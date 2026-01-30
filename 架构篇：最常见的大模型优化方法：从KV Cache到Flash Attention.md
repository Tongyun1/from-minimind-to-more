# 最常见的大模型优化方法：从KV Cache到Flash Attention

## 写在前面

本文基于Minimind使用到的优化技术，对 **KV Cache**、**Attention 变体（MQA/GQA）** 以及 **Flash Attention** 进行了梳理和总结。由于大模型的优化是一个庞大而复杂的技术体系，Minimind仅仅使用了其中最简单常用的一部分。若读者希望更深入地了解这些技术，可以参考另一篇文章《大规模语言模型推理与训练优化机制》。此外，也推荐阅读Jay Alammar大佬的文章https://jalammar.github.io/illustrated-gpt2/。

## 1. 摘要

随着大语言模型（LLM）参数规模从数十亿（7B）向数千亿（70B-400B+）乃至万亿级别跃升，以及上下文窗口从标准的4k token扩展至128k甚至1M token，深度学习的计算重心已显著从模型训练阶段的算力（FLOPS）瓶颈，转移至推理阶段的显存带宽（Memory Bandwidth）与显存容量（Memory Capacity）瓶颈。在自回归（Autoregressive）生成范式下，传统的Transformer架构面临着计算效率与存储效率的双重挑战。

本文解读了Minimind所涉及的核心优化技术：**KV Cache（键值缓存）**、**Multi-Head Attention（MHA，多头注意力）**、**Multi-Query Attention（MQA，多查询注意力）**、**Grouped-Query Attention（GQA，分组查询注意力）**以及**Flash Attention（闪电注意力）**并进行详尽的理论剖析。此外，我还写了一个**例题章节**，供读者练习显存计算。

KV Cache是实现实时自回归生成的必要前提，但其引入的显存线性增长问题迫使架构设计从MHA向MQA和GQA转型，以在模型精度与推理吞吐量（Throughput）之间寻找最优平衡。与此同时，Flash Attention通过算法层面的IO感知（IO-Aware）优化，打破了传统注意力机制的二次方复杂度存储瓶颈，成为长文本推理的基石。

## 2. 自回归推理的物理约束与计算瓶颈

要深刻理解为何需要KV Cache、GQA及Flash Attention，首先必须从底层物理层面解构大语言模型的推理过程。与BERT等编码器（Encoder-only）模型的并行处理不同，基于GPT的解码器（Decoder-only）架构采用自回归生成模式，即下一个Token的生成严格依赖于所有历史Token。

### 2.1 算术强度与内存墙（Memory Wall）

在高性能计算领域，**Roofline模型**定义了硬件性能的理论上限。计算性能 $P$ 受限于峰值算力 $\pi$ 和峰值内存带宽 $\beta$：

$$P = \min(\pi, I \times \beta)$$

其中 $I$ 为**算术强度（Arithmetic Intensity）**，定义为每字节内存访问所进行的浮点运算次数（FLOPs/Byte）。

- **预填充阶段（Prefill Phase）**：当模型处理用户输入的Prompt时，由于所有Token并行计算，Attention与MLP层的矩阵乘法（GEMM）具有较高的算术强度，此时主要受限于GPU的Tensor Core算力（Compute Bound）。
- **解码阶段（Decoding Phase）**：在逐个生成Token的阶段，模型必须为每一个新Token加载全部几百GB的模型权重，却仅进行一次向量-矩阵乘法（GEMV）。这意味着算术强度极低，硬件性能完全受限于显存带宽（Memory Bound）。

例如，对于一个70B参数的模型（FP16精度约140GB），在H100 GPU（3.35TB/s带宽）上，理论最大推理速度仅为：

$$\text{Speed} = \frac{3350 \text{ GB/s}}{140 \text{ GB}} \approx 24 \text{ Tokens/s}$$

这与批处理大小（Batch Size）无关，揭示了**内存墙**是制约LLM推理速度的根本物理障碍。

如图所示，在解码器的自回归生成过程中，给定一个输入，模型预测下一个词元，然后在下一步中将组合输入进行下一次预测。

![img](https://miro.medium.com/v2/resize:fit:1400/0*sexO6adGhaKr7aH0.gif)

这种自回归行为会重复一些操作，我们可以通过放大解码器中计算的掩码缩放点积注意力计算来更好地理解这一点。

![img](https://miro.medium.com/v2/resize:fit:1400/1*8xqD4AYTwn6mQXNw0uhDCg.gif)

由于解码器是因果的（即，一个标记的注意力仅取决于其前面的标记），因此在每个生成步骤中，我们都在重新计算同一个先前标记的注意力，而我们实际上只想计算新标记的注意力。

### 2.2 注意力机制的二次方困境

标准Transformer的核心是缩放点积注意力（Scaled Dot-Product Attention）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于长度为 $L$ 的序列，计算 $QK^T$ 会生成一个 $L \times L$ 的注意力分数矩阵。

- **时间复杂度**：$O(L^2)$
- **空间复杂度**：$O(L^2)$

随着上下文长度 $L$ 的增加（如从4k增加到128k），中间激活值的显存占用呈二次方爆炸式增长。传统的Attention算子需要将庞大的注意力矩阵写入GPU的HBM（高带宽内存），然后再读回进行Softmax计算，这种频繁的读写操作不仅耗时，且极易导致显存溢出（OOM）。

## 3. KV Cache：自回归推理的基石

**KV Cache（键值缓存）** 是大模型推理中最基础的优化技术，其核心目的是以空间换时间，消除自回归生成中的冗余计算。

### 3.1 运行机制详解

在Transformer的每一层中，输入Token经过投影矩阵 $W_Q, W_K, W_V$ 变换为查询（Query）、键（Key）和值（Value）向量。

#### 无缓存推理（Inefficient Decoding）

假设当前已生成序列为 "The sun rises"，需要预测下一个词。

1. 输入 "The sun rises"。
2. 模型重新计算 "The"、"sun"、"rises" 的所有 $Q, K, V$ 向量。
3. 计算注意力并输出 "in"。 这种方法的问题在于，"The"、"sun" 的 $K, V$ 向量在之前的步骤中已经计算过，且对于固定的历史上下文，它们是不会改变的 。

#### 有缓存推理（Efficient Decoding with KV Cache）

1. **首Token生成（Prefill）**：计算Prompt中所有Token的 $K, V$，并将其存入显存中的特定区域（Cache）。
2. **后续生成（Decode）**：当生成第 $n$ 个Token时，仅需计算当前Token的 $q_n, k_n, v_n$。
3. **拼接与检索**：将 $k_n, v_n$ 追加到缓存末尾。注意力计算变为 $q_n$ 与 $K_{cache} + k_n$ 的交互。

通过缓存之前的键值对，我们可以专注于计算新token的注意力。

![img](https://miro.medium.com/v2/resize:fit:1400/1*uyuyOW1VBqmF5Gtv225XHQ.gif)

### 3.2 为什么不缓存 Query (Q)？

这是一个常见的误解。在注意力机制中，Query代表“当前关注点”，它随着生成的推进而不断变化。例如，在生成第10个词时，我们关注的是第10个词与前9个词的关系；生成第11个词时，我们关注的是第11个词与前10个词的关系。因此，Query向量在每一步都是全新的，必须重新计算。而历史Token作为被关注的对象（Key）和信息载体（Value），其特征是固定的，因此可以被缓存 。

### 3.3 KV Cache 的显存挑战

虽然KV Cache将计算复杂度从 $O(L^2)$ 降低到了 $O(L)$（线性扫描历史），但它引入了巨大的显存开销。缓存的大小与以下因素成正比：

- 层数（Layers）
- 批处理大小（Batch Size）
- 序列长度（Sequence Length）
- 注意力头数（Heads）与维度（Head Dimension）

公式如下：

$$M_{KV} = 2 \times N_{layers} \times N_{heads} \times D_{head} \times L_{seq} \times B_{batch} \times P_{size}$$

随着Batch Size的增加（为了提高吞吐量）和Sequence Length的延长（长文档分析），KV Cache的体积往往会超过模型权重本身，成为限制并发量的主要瓶颈。这直接催生了注意力架构的演进。

## 4. 注意力架构的演进：MHA、MQA 与 GQA

为了缓解KV Cache带来的显存压力，研究界提出了一系列改进的注意力结构。这一演进过程本质上是在**模型表现力（Quality）**与**显存效率（Efficiency）**之间寻找最佳平衡点。

![img](https://camo.githubusercontent.com/96963fe92e1c743b7ede42513f411f36e7fd0f9ff12a6fa7108283a33628117d/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f696d675f636f6e766572742f32663663626566383332333964616335633538383937643736346139646262332e776562703f782d6f73732d70726f636573733d696d6167652f666f726d61742c706e67)

### 4.1 Multi-Head Attention (MHA)：多头注意力

**定义**：MHA是《Attention Is All You Need》论文中提出的原始架构 。

- **结构**：拥有 $H$ 个Query头，$H$ 个Key头，和 $H$ 个Value头。
- **比例**：Query : Key : Value = $H : H : H$。
- **特点**：每个Query头都有其专属的Key和Value空间，能够独立地捕捉输入序列中不同的语义特征（如语法结构、指代关系等）。
- **缺点**：推理时显存占用极大。对于每一个Query头，都需要加载对应的Key/Value缓存，导致显存带宽压力巨大，严重限制了最大Batch Size。
- **代表模型**：GPT-3, Llama 1, Llama 2 (7B/13B)。

### 4.2 Multi-Query Attention (MQA)：多查询注意力

**定义**：MQA由Noam Shazeer在2019年提出 ，是一种极端的显存优化方案。

- **结构**：保留 $H$ 个Query头，但所有Query头**共享**同一个Key头和同一个Value头。
- **比例**：Query : Key : Value = $H : 1 : 1$。
- **显存优化**：KV Cache的大小直接缩小了 $H$ 倍。如果模型有32个头，KV Cache仅为MHA的1/32。
- **优势**：
  1. **极大提升吞吐量**：由于需要从HBM加载的数据量大幅减少，算术强度显著提高，使得GPU计算单元能被更充分利用。
  2. **支持超大Batch**：节省的显存可以用于增加Batch Size。
- **劣势**：
  1. **精度损失**：由于所有Query头只能在同一个Key-Value子空间中进行注意力检索，模型的表达能力受到限制，容易导致生成质量下降。
  2. **训练不稳定性**：早期实验表明MQA在训练初期容易出现不收敛的情况 。
- **代表模型**：Falcon-7B/40B, PaLM。

### 4.3 Grouped-Query Attention (GQA)：分组查询注意力

**定义**：GQA是Google Research在2023年提出的折中方案 ，现已成为主流LLM的标配。

- **结构**：将Query头分成 $G$ 个组（Group），每组包含 $H/G$ 个Query头。每个组共享一个Key头和一个Value头。
- **比例**：Query : Key : Value = $H : G : G$。
- **机制**：GQA可以看作是MHA和MQA的泛化形式。当 $G=H$ 时，它就是MHA；当 $G=1$ 时，它就是MQA。通常选取 $G=8$ 作为一个甜点（Sweet Spot）。
- **优势**：
  1. **精度接近MHA**：通过保留一定数量的KV头（如8个），模型仍能捕捉多样的语义特征，性能损失微乎其微。
  2. **速度接近MQA**：显存占用通常降至MHA的1/4或1/8（取决于分组数），大幅降低了显存带宽需求，允许更大的Batch Size。
  3. **平滑升级**：可以通过“Up-training”技术，将现有的MHA模型通过少量计算转换为GQA模型 。
- **代表模型**：Llama 2 (70B), Llama 3 (全系列), Mistral 7B, DeepSeek。

注意，GQA和MQA中，虽然KV的头数较少，但是Minimind会使用repeat_kv()函数来重复它们，再和Q相乘进行注意力计算。

### 4.4 架构对比总结表

下表总结了三种架构在关键指标上的差异（假设基准模型有32个头）：

| **特性指标**       | **Multi-Head Attention (MHA)** | **Grouped-Query Attention (GQA)** | **Multi-Query Attention (MQA)** |
| ------------------ | ------------------------------ | --------------------------------- | ------------------------------- |
| **KV头数量**       | 32 (与Query相等)               | 8 (典型值，Query的1/4)            | 1                               |
| **KV Cache大小**   | 基准 (100%)                    | 25% (显著减少)                    | ~3% (极小)                      |
| **显存带宽压力**   | 极高 (瓶颈)                    | 中等                              | 最低                            |
| **最大Batch Size** | 小                             | 大                                | 极大                            |
| **模型精度**       | 最高                           | 极接近MHA                         | 有明显损失                      |
| **推理延迟**       | 高                             | 低                                | 最低                            |
| **适用场景**       | 小参数模型 (7B以下)            | 主流大模型 (70B+)                 | 极端追求速度的场景              |

## 5. Flash Attention：IO层面的优化

虽然GQA减少了KV Cache的**存储量**，但Flash Attention解决的是注意力计算过程中的**数据传输效率**问题。这是一种从GPU硬件特性出发的系统级优化。

### 5.1 显存层级与IO瓶颈

GPU的内存结构呈金字塔型：

1. **SRAM (Shared Memory/L1 Cache)**：速度极快（19TB/s+），但容量极小（每SM仅100KB-200KB）。
2. **HBM (High Bandwidth Memory)**：速度较慢（2-3TB/s），容量大（40GB-80GB）。

在标准Attention实现中，计算 $S = QK^T$ 后，会将巨大的矩阵 $S$ ($N \times N$) 写入HBM。接着从HBM读取 $S$ 进行Softmax计算，结果 $P$ 再写回HBM。最后再读取 $P$ 和 $V$ 计算 $O = PV$。这种频繁的HBM读写（Read/Write）占据了大部分运行时间，而非矩阵乘法本身 。

### 5.2 Flash Attention V1 & V2 核心原理

Flash Attention提出了一种**IO感知（IO-Aware）**算法，通过**分块（Tiling）**和**重计算（Recomputation）**技术，将整个Attention操作融合为一个CUDA Kernel。

#### 5.2.1 分块（Tiling）

算法将输入的 $Q, K, V$ 矩阵切分成能放入SRAM的小块。在SRAM内部完成矩阵乘法、Softmax缩放等操作。

- **关键创新**：利用Softmax的数学性质，在分块计算过程中动态更新归一化因子（Normalization Statistics），从而无需访问全量矩阵即可计算出正确的局部结果。
- **结果**：中间巨大的 $N \times N$ 注意力矩阵从未完整地写入过HBM，直接在SRAM中“流转”并被消耗，最终只输出 $N \times d$ 的结果到HBM。

#### 5.2.2 重计算（Recomputation）

在训练的反向传播阶段，为了计算梯度，通常需要前向传播时的注意力矩阵。由于Flash Attention不存储该矩阵，它选择在反向传播时**重新计算**一遍前向过程。

- **反直觉结论**：虽然计算量（FLOPs）增加了，但由于减少了极慢的HBM访问，总的墙钟时间（Wall-clock time）反而大幅缩短（加速2-4倍）。

#### 5.2.3 Flash Attention V2 的改进

V2版本  进一步优化了并行策略：

- **序列长度并行**：V1主要在Batch和Head维度并行，V2引入了序列维度的并行，这对于Long Context训练至关重要。
- **循环顺序调整**：V2将外层循环改为在这个Query块上迭代K/V块，减少了写入HBM的同步开销。

### 5.3 Flash Decoding：推理专用的优化

Flash Attention主要优化的是训练和预填充（Prefill）阶段。对于生成阶段（Decoding），由于Query长度通常为1，Batch Size较小时，GPU利用率极低。

**Flash Decoding**  专门针对此场景：

- **Split-K 并行**：将长序列的KV Cache切分成多个块（Chunks）。
- **并行计算**：启动多个CUDA Thread Block并行计算Query与这些KV Chunks的注意力分数。
- **归约（Reduction）**：最后通过一个Reduce操作合并各块的结果。
- **效果**：在长上下文（如32k+）推理时，Flash Decoding能将速度提升8倍以上，因为它充分利用了GPU的所有SM核心，而不是让大部分核心空闲。

## 6. 显存占用计算与数学框架

为了量化上述技术的影响，我们建立一个通用的显存计算框架。这对于硬件选型和部署规划至关重要。

### 6.1 核心参数定义

| **符号**     | **含义**                 | **单位/说明**                    |
| ------------ | ------------------------ | -------------------------------- |
| $P_{model}$  | 模型参数量               | Billions ($10^9$)                |
| $L_{seq}$    | 上下文序列长度           | Tokens                           |
| $B$          | 批处理大小 (Batch Size)  | 整数                             |
| $D_{model}$  | 隐藏层维度 (Hidden Size) | 整数                             |
| $N_{layers}$ | 模型层数                 | 整数                             |
| $N_{heads}$  | Query注意力头数          | 整数                             |
| $N_{kv}$     | KV注意力头数             | MHA时等于 $N_{heads}$，GQA时更小 |
| $D_{head}$   | 单头维度                 | 通常 $D_{model} / N_{heads}$     |
| $S_{prec}$   | 数据精度大小             | Bytes (FP16=2, FP32=4, INT8=1)   |

### 6.2 显存占用公式

总显存占用 $M_{total}$ 由两部分组成：

$$M_{total} = M_{weights} + M_{kv\_cache} + M_{activation}$$

*(注：推理阶段激活值 $M_{activation}$ 相对较小，主要由权重和KV Cache主导，但在长序列下KV Cache占主导)*

#### 1. 模型权重显存 ($M_{weights}$)

$$M_{weights} \approx P_{model} \times S_{prec}$$

#### 2. KV Cache 显存 ($M_{kv\_cache}$)

$$M_{kv\_cache} = 2 \times N_{layers} \times N_{kv} \times D_{head} \times L_{seq} \times B \times S_{prec}$$

*系数“2”代表同时存储Key和Value。*

## 7. 综合计算例题（Case Studies）

为了方便理解，我们在开始计算前先统一两个核心公式（基于FP16精度，每个参数占2 Bytes）：

1. **静态显存（模型权重）**：

   $$M_{weight} = \text{参数量} \times 2$$

2. **动态显存（KV Cache）**：

   $$M_{kv} = 2 \times \text{层数} \times \text{KV头数} \times \text{头维} \times \text{序列长} \times \text{Batch} \times 2$$

------

### 例题 1：Llama 2 7B (MHA) vs Llama 3 8B (GQA)

**—— 为什么新一代模型更省显存？**

**背景**：

我们需要对比两代模型在处理相同任务时（Batch Size=16, 长度=4096）的显存占用，看能否放入一张 RTX 4090 (24GB)。

**模型参数对比**：

- **Llama 2 7B (旧架构)**: 32层, **32个KV头 (MHA)**, 隐藏层维度128。
- **Llama 3 8B (新架构)**: 32层, **8个KV头 (GQA)**, 隐藏层维度128。

#### 第一步：计算静态权重显存

这是模型加载后雷打不动的显存占用。

- **Llama 2 7B**: $7 \times 10^9 \times 2 \text{ Bytes} \approx \mathbf{14 \text{ GB}}$

- **Llama 3 8B**: $8 \times 10^9 \times 2 \text{ Bytes} \approx \mathbf{16 \text{ GB}}$

  *(注：Llama 3 参数略多，所以基础占用其实更高)*

#### 第二步：计算动态 KV Cache (Llama 2 7B)

MHA 架构下，KV 头数 = Attention 头数 = 32。

$$\begin{aligned} M_{kv} &= 2 \times 32(\text{层}) \times \mathbf{32(\text{KV头})} \times 128(\text{维}) \times 4096(\text{长}) \times 16(\text{Batch}) \times 2(\text{Bytes}) \\ &= 34,359,738,368 \text{ Bytes} \end{aligned}$$

**换算为 GB**: $34,359,738,368 \div 1024^3 \approx \mathbf{32 \text{ GB}}$

> **Llama 2 结果**：
>
> 总需求 = 14 GB (权重) + 32 GB (KV Cache) = **46 GB**。
>
> **结论**：远远超过 RTX 4090 (24GB)，无法运行。

#### 第三步：计算动态 KV Cache (Llama 3 8B)

GQA 架构下，KV 头数减少为 8（显存优化的关键）。

$$\begin{aligned} M_{kv} &= 2 \times 32(\text{层}) \times \mathbf{8(\text{KV头})} \times 128(\text{维}) \times 4096(\text{长}) \times 16(\text{Batch}) \times 2(\text{Bytes}) \\ &= 8,589,934,592 \text{ Bytes} \end{aligned}$$

**换算为 GB**: $8,589,934,592 \div 1024^3 \approx \mathbf{8 \text{ GB}}$

> **Llama 3 结果**：
>
> 总需求 = 16 GB (权重) + 8 GB (KV Cache) = **24 GB**。
>
> **结论**：刚好填满 RTX 4090。虽然很极限（实际需要留一点余量给系统），但相比 Llama 2，显存占用减少了近一半，使得个人显卡跑大 Batch 成为可能。

------

### 例题 2：Llama 2 70B 的并发极限

**—— 手里的显卡到底能支持多少人同时用？**

**背景**：

公司有 2 张 A100 (80GB)，共 160GB 显存。要部署 Llama 2 70B 模型，每个用户的上下文长度为 4096。请问最多能支持多少人同时请求（Max Batch Size）？

*(Llama 2 70B 采用了 GQA 技术，KV头数=8，层数=80)*

#### 第一步：计算剩余可用显存

显存是一个“大桶”，必须先装入模型权重，剩下的空间才能用来服务用户（存 KV Cache）。

1. **总显存**：$80 \text{ GB} \times 2 = 160 \text{ GB}$。

2. **权重占用**：$70 \times 10^9 \times 2 \text{ Bytes} \approx \mathbf{140 \text{ GB}}$。

   *(注：这里为了保险起见，预留一些空间给 PyTorch 上下文和激活值，我们假设实际可用给 KV Cache 的空间约为 **15 GB**。)*

#### 第二步：计算“单个用户”的 KV 开销

我们先算 **Batch Size = 1** 时的 KV Cache 大小。

$$\begin{aligned} M_{single} &= 2 \times 80(\text{层}) \times 8(\text{KV头}) \times 128(\text{维}) \times 4096(\text{长}) \times 1(\text{Batch}) \times 2(\text{Bytes}) \\ &= 1,342,177,280 \text{ Bytes} \end{aligned}$$

**换算为 GB**: $1.34 \times 10^9 \div 1024^3 \approx \mathbf{1.25 \text{ GB}}$。

这意味着：每进来一个用户，就要消耗 1.25 GB 显存。

#### 第三步：计算最大 Batch Size

$$B_{max} = \frac{\text{剩余可用显存}}{\text{单用户开销}} = \frac{15 \text{ GB}}{1.25 \text{ GB}} = 12$$

> **结论**：
>
> 在 2 张 A100 上，你最多只能同时服务 **12** 个并发用户。
>
> **对比思考**：如果 70B 模型没有使用 GQA（假设用 MHA，KV头数是 64 而不是 8），单用户开销会变成 $1.25 \times 8 = 10 \text{ GB}$。那么 $15 / 10 = 1.5$，也就是连 **2 个并发**都跑不起来。这就体现了 GQA 对商业落地的决定性意义。

------

### 例题 3：超长上下文 (128k) 的瓶颈分析

**—— 为什么长文本这么慢？**

**背景**：

我们要让 Llama 3 8B 处理一本小说（128k tokens, 约 13万字）。Batch Size = 1。

#### 第一步：计算 KV Cache 大小

将序列长度 $L$ 替换为 131,072 (128k)。

$$\begin{aligned} M_{kv} &= 2 \times 32 \times 8 \times 128 \times \mathbf{131072} \times 1 \times 2 \\ &= 17,179,869,184 \text{ Bytes} \approx \mathbf{16 \text{ GB}} \end{aligned}$$

**显存现状**：权重 16GB + KV Cache 16GB = 32GB。

**结论 1**：单张 4090 (24GB) 即使跑 Batch=1 也会爆显存。必须使用 **4-bit 量化**（将权重压缩到 5-6GB）才能运行。

#### 第二步：分析带宽瓶颈（Memory Bound）

这是最难理解但最重要的部分。

当模型生成第 128,001 个 token 时，GPU 需要回顾前面所有的 128,000 个 token 的 KV 信息来做一次 Attention 计算。

- **数据搬运量**：需要从显存读取 **16 GB** 的数据进入计算核心。

- **硬件能力**：RTX 4090 的显存带宽约为 **1 TB/s** (1000 GB/s)。

- **物理耗时**：

  $$T_{io} = \frac{\text{数据量}}{\text{带宽}} = \frac{16 \text{ GB}}{1000 \text{ GB/s}} = 0.016 \text{ 秒} = \mathbf{16 \text{ ms}}$$

这仅仅是读取数据的时间，还没开始算！

这意味着：每生成**一个字**，GPU 都要花 16ms 去“搬砖”。对于人眼来说，这会造成明显的延迟感。

#### 第三步：Flash Attention / Decoding 的作用

如果不优化，GPU 就像用一根吸管（单线程）去吸这 16GB 的水，速度非常慢。

- **Flash Decoding** 的技术原理是将这 128k 的数据**切分**成很多小块（Split-KV），分发给 GPU 上所有的计算单元（SM）同时读取和计算。
- 它虽然不能减少 16GB 的物理大小，但能极大提升读取和计算的**并行度**，避免 GPU 核心空转等待数据。

## 8. 未来展望

### 8.1 MLA

DeepSeek-V2/V3 引入了 **Multi-Head Latent Attention (MLA)** 。MLA 通过低秩矩阵分解技术，将 Key-Value 压缩为一个极小的 Latent Vector。

- **对比 GQA**：GQA 只是减少了头的数量（$N_{kv}$），而 MLA 压缩了每个头内部的数据表示。
- **效果**：DeepSeek-V2 (236B) 的 KV Cache 甚至比 Llama 3 70B (GQA) 还要小。这使得万亿参数模型的推理在显存成本上变得可接受。

### 8.2 PagedAttention

在软件层面，**vLLM** 库引入的 **PagedAttention** 解决了KV Cache在显存中物理连续存储导致的碎片化问题。类似于操作系统的虚拟内存分页，PagedAttention 允许 KV Block 离散存储，将显存利用率从 60% 提升至 95% 以上，与 GQA 配合形成了现代推理栈的标准配置。



通过 GQA 和 Flash Attention，我们成功驯服了显存与计算的复杂度。然而，传统的 Dense FFN 依然要求每次推理都‘全员出动’激活所有参数，这无疑是一种巨大的算力浪费。 **在下一篇文章中，我们将介绍近期最火的MoE架构。我们将把手术刀对准 FFN 层，剖析 SwiGLU 与 Hybrid MoE 的结合，探究MiniMind 仅在需要时激活特定的‘神经元专家’，在保持极小显存占用的同时，获得超越量级的知识容量**

## 9. 动手实践：Minimind代码解析

以下是https://github.com/jingyaogong/minimind 的model/model_minimind.py中的attention部分，我进行了详细注释

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 Key/Value heads 以实现 GQA (Grouped Query Attention)
    
    GQA 是一种注意力机制优化，使用较少的 KV heads 来匹配更多的 Query heads。
    例如：8 个 Query heads 对应 2 个 KV heads，每个 KV head 需要重复 4 次。
    
    这样可以减少 KV 缓存的大小，在推理时节省显存。
    
    Args:
        x: Key 或 Value 张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 每个 KV head 需要重复的次数（n_rep = num_heads / num_kv_heads）
        
    Returns:
        重复后的张量 [batch, seq_len, num_heads, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    
    # 如果不需要重复（n_rep=1），直接返回
    if n_rep == 1:
        return x
    
    # 在维度 3 插入新维度，然后扩展并重塑
    # 例如：[B, L, 2, D] -> [B, L, 2, 1, D] -> [B, L, 2, 4, D] -> [B, L, 8, D]
    return (
        x[:, :, :, None, :]  # [B, L, num_kv_heads, 1, head_dim]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # [B, L, num_kv_heads, n_rep, head_dim]
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # [B, L, num_heads, head_dim]
    )

class Attention(nn.Module):
    """
    多头注意力机制（支持 GQA 和 Flash Attention）
    
    实现了标准的缩放点积注意力（Scaled Dot-Product Attention），支持：
    1. GQA (Grouped Query Attention): 使用较少的 KV heads 匹配更多的 Query heads
    2. Flash Attention: 使用 PyTorch 2.0+ 的优化注意力实现
    3. RoPE: 通过旋转位置编码将位置信息注入 Q 和 K
    4. KV Cache: 支持推理时的 KV 缓存加速
    
    注意力公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, args: MiniMindConfig):
        """
        初始化注意力层
        
        Args:
            args: MiniMindConfig 配置对象
        """
        super().__init__()
        # ========== GQA 配置 ==========
        # num_key_value_heads: KV heads 数量（通常小于 Query heads）
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 确保 Query heads 数量能被 KV heads 数量整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads  # Query heads 数量
        self.n_local_kv_heads = self.num_key_value_heads  # KV heads 数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV head 需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度
        
        # ========== 投影层 ==========
        # Q 投影：hidden_size -> num_heads * head_dim
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # K 投影：hidden_size -> num_kv_heads * head_dim（GQA：较少的 heads）
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # V 投影：hidden_size -> num_kv_heads * head_dim（GQA：较少的 heads）
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：num_heads * head_dim -> hidden_size
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # ========== Dropout ==========
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力分数 dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接 dropout
        self.dropout = args.dropout
        
        # ========== Flash Attention ==========
        # 检查是否支持 Flash Attention（需要 PyTorch >= 2.0）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
            past_key_value: 缓存的 KV 值，用于增量解码 [batch, past_len, num_kv_heads, head_dim]
            use_cache: 是否返回 KV 缓存供下次使用
            attention_mask: 注意力掩码 [batch, seq_len]，1 表示有效位置，0 表示掩码位置
            
        Returns:
            output: 注意力输出 [batch, seq_len, hidden_size]
            past_kv: 新的 KV 缓存（如果 use_cache=True），否则为 None
        """
        bsz, seq_len, _ = x.shape
        
        # ========== 步骤 1：Q/K/V 投影 ==========
        # 将输入投影到 Q、K、V 空间
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 重塑为多头格式：[batch, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # GQA：较少的 KV heads
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # GQA：较少的 KV heads
        
        # ========== 步骤 2：应用 RoPE 位置编码 ==========
        # 将位置信息编码到 Q 和 K 中
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        
        # ========== 步骤 3：KV Cache 处理 ==========
        # 如果有缓存的 KV 值（增量解码），将其与当前 KV 拼接
        if past_key_value is not None:
            # past_key_value[0] 是缓存的 K，past_key_value[1] 是缓存的 V
            # 在序列维度（dim=1）上拼接：[batch, past_len+seq_len, num_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 如果需要缓存，保存当前的 KV 值
        past_kv = (xk, xv) if use_cache else None
        
        # ========== 步骤 4：GQA 处理 ==========
        # 调整维度顺序为 [batch, num_heads, seq_len, head_dim]（Flash Attention 格式）
        # 对于 KV，需要重复 heads 以匹配 Query heads 数量
        xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        
        # ========== 步骤 5：计算注意力 ==========
        # 优先使用 Flash Attention（如果支持且条件满足）
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention：使用 PyTorch 优化的注意力实现
            # 条件：序列长度 > 1，没有 KV cache，没有复杂掩码
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自动应用因果掩码
            )
        else:
            # 标准注意力计算
            # 步骤 5.1：计算注意力分数 QK^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_heads, seq_len, kv_len]
            
            # 步骤 5.2：应用因果掩码（只对当前序列部分）
            # 上三角矩阵掩码，防止看到未来的 token
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            )
            
            # 步骤 5.3：应用注意力掩码（如果有）
            if attention_mask is not None:
                # 将掩码扩展到 [batch, 1, 1, seq_len] 并转换为分数掩码
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # 0 -> -inf, 1 -> 0
                scores = scores + extended_attention_mask
            
            # 步骤 5.4：Softmax 归一化
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            # 步骤 5.5：应用 dropout
            scores = self.attn_dropout(scores)
            
            # 步骤 5.6：加权求和
            output = scores @ xv  # [batch, num_heads, seq_len, head_dim]
        
        # ========== 步骤 6：输出投影 ==========
        # 重塑并投影回 hidden_size
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [batch, seq_len, num_heads * head_dim]
        output = self.resid_dropout(self.o_proj(output))  # [batch, seq_len, hidden_size]
        
        return output, past_kv
```

