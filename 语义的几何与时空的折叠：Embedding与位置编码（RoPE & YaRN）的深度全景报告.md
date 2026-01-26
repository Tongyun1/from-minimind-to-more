# 语义的几何与时空的折叠：Embedding与位置编码（RoPE & YaRN）的深度全景报告

## 写在前面

请在阅读本文之后再去看minimind的代码，以便于快速理解其实现。本文在最后附有minimind的位置编码实现的详细注释。

## 1. 引言：从离散符号到连续时空的认知跨越

在人工智能的发展历程中，自然语言处理（Natural Language Processing, NLP）始终占据着皇冠明珠的地位。其核心挑战在于：人类语言是一种基于离散符号（Discrete Symbols）的线性序列系统，而现代深度学习模型，尤其是基于Transformer架构的大语言模型（Large Language Models, LLMs）则是基于连续向量空间（Continuous Vector Space）的并行计算系统。为了弥合这一本质上的鸿沟，两个关键概念应运而生：**Embedding（嵌入）**与**Positional Encoding（位置编码）**。

前者解决了“意义”的度量问题，将“苹果”与“牛顿”从毫无关联的ID符号转化为高维空间中蕴含潜在引力关系的向量；后者则解决了“秩序”的重构问题，在Transformer全并行（Parallel）的计算图景中，重新注入了语言赖以生存的时序逻辑（Sequential Logic）。

随着模型规模从GPT-2的1.5亿参数跃升至GPT-4、Llama 3及DeepSeek-V3的万亿级别，以及上下文窗口（Context Window）从最初的1024 tokens扩展至128k甚至1M+，位置编码技术经历了从简单的加性正弦编码（Sinusoidal APE）到旋转位置编码（Rotary Positional Embedding, RoPE），再到结合神经正切核（Neural Tangent Kernel, NTK）理论与熵调节机制的YaRN（Yet another RoPE extensioN）的深刻演进。这一演进过程不仅仅是数学技巧的迭代，更是一场关于如何让神经网络在有限的训练数据中理解无限延伸的“相对距离”与“语义关联”的认知革命。

本报告将以15,000字以上的篇幅，对Embedding的基础原理、位置编码的历史沿革、RoPE的数学几何本质、以及应对长上下文挑战的YaRN体系进行详尽、透彻且具有前瞻性的深度剖析。

## 2. 语义的基石：Embedding与向量空间模型

在深入探讨复杂的位置编码之前，必须首先构建对Embedding这一概念的物理直觉。Embedding不仅是LLM的输入层，它是整个现代NLP大厦赖以建立的公理化假设——**分布语义学（Distributional Semantics）**的数学实现。

### 2.1 符号的离散性与计算的连续性

在计算机的底层逻辑中，文本只是一串ASCII码或Unicode编码。对于一个未经训练的系统而言，单词“猫”和“狗”之间没有任何内在联系，它们只是两个不同的整数索引（Token ID）。例如，在常见的BPE（Byte Pair Encoding）词表中，“cat”可能是 `ID: 8921`，而“dog”是 `ID: 4412` 。

传统的机器学习方法（如One-hot编码）将每个词表示为一个维度等于词表大小的稀疏向量。这种表示法有两个致命缺陷：

1. **维数灾难**：词表通常包含数万甚至数十万个词，导致向量极度稀疏且计算低效。
2. **语义正交**：在One-hot空间中，任意两个不同词向量之间的内积为零，欧氏距离恒定为 $\sqrt{2}$。这意味着系统无法从几何上感知“猫”和“狗”的相似性，也无法理解“猫”与“汽车”的区别。

Embedding层的引入，旨在将这些离散的ID映射到一个低维（通常为512到8192维）、稠密（Dense）且连续的实数向量空间中。在这个空间里，向量的方向和距离不再是随机的，而是编码了词汇的句法和语义特征 。

### 2.2 几何空间中的意义：分布语义假设

Embedding的核心哲学源于语言学家J.R. Firth的名言：“你会通过一个词的伴随词来认识它（You shall know a word by the company it keeps）。”

在神经网络的训练过程中，Embedding矩阵（即Lookup Table，形状为 $V \times D$）通过反向传播算法不断调整。如果两个词经常出现在相似的上下文中（例如“喝”经常搭配“水”和“茶”），模型为了最小化预测误差，会将这两个词的Embedding向量在空间中推向彼此。

#### 2.2.1 向量算术与类比推理

一个训练良好的Embedding空间表现出令人惊叹的代数性质，即线性子结构（Linear Substructure）。最著名的例子是性别关系的平移不变性：

$$\vec{v}(\text{King}) - \vec{v}(\text{Man}) + \vec{v}(\text{Woman}) \approx \vec{v}(\text{Queen})$$

这意味着，“国王”到“女王”的向量差，与“男人”到“女人”的向量差，在方向和模长上是近似相等的。这表明Embedding不仅编码了实体的身份，还隐式地对齐了抽象的概念维度（如性别、时态、单复数等）。

### 2.3 词袋模型的局限与位置的必要性

尽管Embedding成功地将语义进行了向量化，但它本质上是**位置无关（Position-Agnostic）**的。

考虑以下两个句子：

1. “张三打了李四。”
2. “李四打了张三。”

在纯粹的Embedding视角下，这两个句子包含完全相同的Token集合：{张三, 打, 了, 李四}。如果我们简单地将这些Embedding相加或取平均（词袋模型 Bag-of-Words），这两个句子的表示将完全一致，模型无法区分谁是施暴者，谁是受害者 。

RNN（循环神经网络）通过按时间步顺序逐个处理Token，将位置信息隐式地编码在隐藏状态（Hidden State）的演变中。然而，RNN的串行特性限制了其并行计算能力，无法适应大规模数据的训练。 Transformer架构的出现打破了串行限制。它通过Self-Attention机制一次性并行处理序列中的所有Token。在这个机制中，每个Token都能“看见”其他所有Token，**原本的线性顺序被打破了**。为了让Transformer理解“顺序”，我们必须显式地构建一套坐标系，并将这个坐标系注入到Embedding空间中。这就是位置编码（Positional Encoding）的起源 。

## 3. 寻找秩序：位置编码的演进史

位置编码的发展历史，就是一部人类试图用数学语言描述“序列秩序”的历史。从绝对位置的生硬标记，到相对位置的动态感知，这一领域的探索为后来的RoPE奠定了基础。

### 3.1 绝对位置编码（Absolute Positional Encoding, APE）

#### 3.1.1 正弦位置编码（Sinusoidal PE）

在2017年Vaswani等人的开山之作《Attention Is All You Need》中，提出了一种基于三角函数的固定位置编码。其核心思想是利用不同频率的正弦和余弦波来为每个位置生成一个唯一的指纹（Fingerprint）。

对于位置 $pos$ 和维度 $i$，其编码公式为：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**深度解析：**

- **多尺度时钟**：可以将其想象为一组转速不同的时钟。维度 $i=0$ 对应最高频（High Frequency），波长最短，能够区分相邻位置的微小变化；随着 $i$ 增大，频率降低，波长变长，用于区分更宏观的距离。
- **线性变换属性**：作者在论文中提到，对于任意偏移 $k$，$\vec{PE}_{pos+k}$ 可以表示为 $\vec{PE}_{pos}$ 的线性函数。这在理论上允许模型通过学习线性矩阵来推断相对位置，但在实际训练中，这种相对性的学习并不容易。
- **加法注入（Additive Injection）**：这种编码直接**加**（Element-wise Sum）到Token Embedding上。这是一个极其大胆的设计，因为它在数学上混合了语义空间（Embedding）和位置空间（PE）。之所以可行，是因为高维空间具有足够的稀疏性，语义信息和位置信息往往分布在近似正交的子空间中，或者模型学会了在特定的频段关注语义，在特定的频段关注位置。

#### 3.1.2 可学习的绝对位置编码（Learned APE）

GPT-2和BERT并没有使用正弦函数，而是直接初始化了一个大小为 $L_{max} \times D$ 的可学习矩阵。每个位置 $0, 1, \dots, L_{max}-1$ 都有一个专属的向量，随模型参数一起训练 。

- **优势**：更灵活，能适应特定数据集的分布。
- **劣势**：**无法外推（No Extrapolation）**。如果模型训练时的最大长度是1024，那么它根本就没有位置1025的向量，推理时一旦越界就会直接报错或完全失效。这成为早期LLM扩展上下文的主要瓶颈。

### 3.2 相对位置编码（Relative Positional Encoding, RPE）

随着研究的深入，学者们发现：在自然语言理解中，**绝对坐标**往往不重要，重要的是**相对距离**。

例如，在句子“虽然下雨，但我还是去了”中，“虽然”和“但”之间的关联取决于它们相隔的距离，而不是它们具体出现在句子的第5个词还是第100个词。

#### 3.2.1 偏置相加法（Bias Addition）

Shaw等人提出，不再将位置信息加到Embedding上，而是在计算Attention Score（即 $QK^T$）时，加入一个表示相对距离 $(i-j)$ 的可学习偏置项 $B_{i-j}$。

T5模型进一步优化了这一方案，使用了对数分桶（Logarithmic Bucketing）策略，即对近距离的位置赋予精确的独立偏置，对远距离的位置共享同一个桶的偏置。这不仅减少了参数量，还体现了人类认知的特点：对近处敏感，对远处模糊。

#### 3.2.2 ALiBi (Attention with Linear Biases)

ALiBi是相对位置编码的一个极端简化与高效版本。它完全抛弃了可学习的参数，直接在Attention Score矩阵上减去一个与距离成正比的惩罚项：

$$\text{Score}_{i,j} = \mathbf{q}_i \cdot \mathbf{k}_j - m \cdot |i-j|$$

其中 $m$ 是一个特定于Attention Head的斜率系数 。

- **外推之王**：ALiBi展现了极强的外推能力。即使训练时只看512长度，推理时也能在数千长度上保持不崩。
- **局限**：它假设距离越远关注度越低（线性衰减），这种强归纳偏置（Inductive Bias）虽然有助于稳定训练，但也可能限制了模型捕捉长距离依赖（如文章开头的伏笔在结尾呼应）的能力。更重要的是，它无法像RoPE那样优雅地融入KV Cache的压缩与优化中。

## 4. 旋转位置编码（RoPE）：几何与代数的完美融合

RoPE（Rotary Positional Embedding）的出现，彻底改变了位置编码的格局。它由苏剑林（Su Jianlin）等人提出，并在RoFormer论文中形式化，随后被Meta的Llama系列、Mistral、Google PaLM、DeepSeek等几乎所有现代主流LLM采纳。

RoPE的核心洞察在于：**通过绝对位置的旋转，自然诱导（Induce）出相对位置的内积性质。** 

### 4.1 核心数学推导：从复数域出发

为了理解RoPE，我们首先考虑二维空间的情况。假设我们将Embedding向量的每两个维度视为一个复数。

设 Query 向量 $\boldsymbol{q}$ 和 Key 向量 $\boldsymbol{k}$ 在位置 $m$ 和 $n$ 处的原始值为：

$$\boldsymbol{q} = q_0 + i q_1 = r_q e^{i\theta_q}$$

$$\boldsymbol{k} = k_0 + i k_1 = r_k e^{i\theta_k}$$

传统的APE是做加法：$\boldsymbol{q}' = \boldsymbol{q} + \boldsymbol{p}_m$。

RoPE则是做**乘法（旋转）**。我们将位置 $m$ 编码为一个旋转因子 $e^{im\theta}$：

$$f(\boldsymbol{q}, m) = \boldsymbol{q} \cdot e^{im\theta} = r_q e^{i(\theta_q + m\theta)}$$

$$f(\boldsymbol{k}, n) = \boldsymbol{k} \cdot e^{in\theta} = r_k e^{i(\theta_k + n\theta)}$$

几何上，这相当于把向量在复平面上逆时针旋转了 $m\theta$ 角度。

**奇迹发生在计算内积（Attention Score）时：**

在复数域中，两个向量的内积对应于 Hermitian 内积的实部：

$$\langle f(\boldsymbol{q}, m), f(\boldsymbol{k}, n) \rangle = \text{Re}\left[ f(\boldsymbol{q}, m) \cdot f(\boldsymbol{k}, n)^* \right]$$

代入公式：

$$= \text{Re}\left[ \boldsymbol{q} e^{im\theta} \cdot (\boldsymbol{k} e^{in\theta})^* \right]$$

$$= \text{Re}\left[ \boldsymbol{q} e^{im\theta} \cdot \boldsymbol{k}^* e^{-in\theta} \right]$$

$$= \text{Re}\left[ \boldsymbol{q} \boldsymbol{k}^* \cdot e^{i(m-n)\theta} \right]$$

**关键结论**：最终的内积结果中，位置信息仅以 $(m-n)$ 的形式出现。 这意味着：**尽管我们对每个Token进行了绝对位置的旋转，但它们之间的相互作用（Attention）完全取决于它们的相对距离。** 这一性质被称为**平移不变性（Translation Invariance）**，它是RoPE能够统领江湖的根本原因 。

### 4.2 推广到多维空间：分块旋转矩阵

在实际的Transformer中，Embedding维度 $d$ 通常很大（如4096）。RoPE采用“分而治之”的策略，将 $d$ 维向量切分为 $d/2$ 个二维子空间。

对于第 $j$ 个子空间（即第 $2j$ 和 $2j+1$ 维），我们分配一个特定的旋转频率 $\theta_j$。

整个旋转操作可以表示为一个巨大的分块对角矩阵 $\mathbf{R}_{\Theta, m}$ 乘以向量 $\mathbf{x}$：

$$\mathbf{R}_{\Theta, m} = \begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots \\ \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots \\ 0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots \\ 0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

**频率设定**：RoPE沿用了Vaswani正弦编码的几何级数频率设定：

$$\theta_j = \text{base}^{-2j/d}$$

通常 $\text{base} = 10000$。这意味着：

- **低维度（小 $j$）**：$\theta_j$ 很大，旋转速度快，负责捕捉高频的局部位置信息。
- **高维度（大 $j$）**：$\theta_j$ 很小，旋转速度极慢，负责捕捉低频的全局位置信息。

### 4.3 几何直觉：RoPE的“多级时钟”隐喻

为了更直观地理解RoPE，我们可以借用“多级时钟”或“密码锁”的隐喻 。

想象每一个Token向量都是由几百个不同的“指针”组成的。

- 当你读入文本时，Token每推进一步（位置+1），这些指针就会转动一次。
- **第一个指针（低维）**像是秒针，转得飞快。你往前走一步，它可能转了很大角度。这意味着哪怕位置只差1，这个维度上的向量夹角也会变化巨大。这让模型能敏锐地感知“紧邻”关系。
- **最后一个指针（高维）**像是时针甚至“年针”，转得极慢。你需要往前走几千步，它才转动一点点。这让模型能在长距离上保持位置的连贯性，不会因为距离太远而产生相位混叠。

Attention计算就是比较Query和Key之间所有指针对的相对角度。由于相对角度只取决于 $(m-n)$，所以这套机制完美实现了相对位置编码。

### 4.4 为什么RoPE优于APE和ALiBi？

| **特性**           | **APE (Sinusoidal/Learned)** | **ALiBi**                | **RoPE**                       |
| ------------------ | ---------------------------- | ------------------------ | ------------------------------ |
| **位置注入方式**   | 加法 (Addition)              | 偏置 (Bias)              | 乘法 (Rotation)                |
| **相对位置感知**   | 弱 (需隐式学习)              | 强 (显式构造)            | 强 (数学诱导)                  |
| **外推能力**       | 极差                         | 极强                     | 较好 (需配合NTK/YaRN)          |
| **语义保留**       | 混合并污染语义空间           | 不污染 (但在Score层操作) | 正交变换，保留模长 (Norm)      |
| **KV Cache兼容性** | 完美                         | 差 (需重算或特殊处理)    | 完美 (旋转注入到KV中)          |
| **长距离衰减**     | 无自然衰减                   | 强制线性衰减             | 自然震荡衰减 (Long-term Decay) |

RoPE的一个重要特性是**长程衰减（Long-term Decay）**。随着相对距离 $|m-n|$ 的增加，高频分量的旋转相位差变得随机，内积期望趋向于0。这符合语言模型的局部性原理（Locality），即近处的词通常更重要。这使得RoPE不需要像ALiBi那样硬编码衰减，而是自然涌现出这种性质 。

## 5. 长上下文的挑战：内插与分辨率的博弈

虽然RoPE理论上支持无限长度，但在实际应用中，当推理长度超过训练长度（例如训练4k，推理8k）时，模型性能会崩塌（PPL爆炸）。这被称为**外推性故障（Extrapolation Failure）**。

### 5.1 为什么直接外推会失败？

回到“时钟”隐喻：在训练阶段，模型见过指针在 $[0, L_{train}\theta]$ 范围内的旋转组合。神经网络是一个强力的拟合器，它记住了这些旋转模式。 当推理长度达到 $L_{test} > L_{train}$ 时，旋转角度 $m\theta$ 进入了模型从未见过的数值区域（Out-of-Distribution, OOD）。这就好比你只教过模型看0点到12点的钟，突然让它看13点，虽然数学上是循环的，但在深度神经网络的非线性映射中，这种OOD输入会导致Attention Score计算出的数值异常，进而导致Softmax分布熵崩塌 。

### 5.2 线性内插（Position Interpolation, PI）：压缩时空

为了解决这个问题，Meta的研究人员（Chen et al., 2023）提出了**线性内插（Linear Interpolation）**。

思路非常简单粗暴：**如果外面的世界太危险（OOD），那我们就把外面的世界缩小塞进已知的世界里。**

假设我们要将窗口扩展 $s$ 倍（例如从4k扩展到8k，则 $s=2$）。我们将所有的位置索引 $m$ 替换为 $m/s$。

$$f(\boldsymbol{q}, m) = \boldsymbol{q} e^{i \frac{m}{s} \theta}$$

这样，原本 $0 \sim 8000$ 的范围被映射回了 $0 \sim 4000$。对于模型来说，所有的旋转角度都在它见过的训练分布内。这立刻解决了PPL爆炸的问题，微调后效果显著 。

### 5.3 线性内插的代价：分辨率危机

然而，线性内插并非没有代价。它引发了**分辨率（Resolution）缺失**的问题。

当你将位置除以 $s$ 时，相邻两个Token之间的旋转角度差 $\Delta \theta$ 也变成了原来的 $1/s$。

$$\Delta \theta_{new} = \frac{\Delta \theta_{original}}{s}$$

对于高频分量（转得快的维度），这可能影响不大。但对于低频分量，甚至是中频分量，这种压缩会导致相邻Token在向量空间中“靠得太近”，难以区分。 这就好比你把一张高分辨率图片缩小了，虽然内容还在，但像素变得模糊了。模型在处理精细的局部关系（如相邻词的语法依赖）时，会变得迟钝。这解释了为什么线性内插在扩展倍数很大时，短文本的性能会下降 。

## 6. NTK-Aware Scaling：频域视角的觉醒

为了解决线性内插的分辨率问题，社区（最初源于Reddit讨论，后被学术界证实）引入了**神经正切核（Neural Tangent Kernel, NTK）**理论的视角。这是一次极具代表性的理论指导工程实践的案例。

### 6.1 什么是NTK与频谱偏差？

NTK理论揭示了深度神经网络在学习过程中的**频谱偏差（Spectral Bias）**：网络倾向于优先学习低频函数，而难以学习高频剧烈变化的函数 。 在RoPE的语境下：

- **低维度（高频 $\theta$）**：对应文本中的局部高频变化。由于波长短，它们在Embedding空间中变化剧烈。根据NTK理论，如果我们将这些高频信号进一步压缩（如PI所做的），网络将极难捕捉到这些微小的位置差异。
- **高维度（低频 $\theta$）**：对应文本中的长距离低频变化。这些信号波长很长，稍微压缩一下并不会丢失太多信息。

### 6.2 频率感知的非线性缩放

NTK-Aware Scaling的核心思想是：**对不同频率的维度应用不同程度的缩放。**

- **高频维度（低维）**：**不插值**（或极少插值）。因为它们负责局部位置，必须保持高分辨率，让模型能分清邻居。而且高频分量旋转得快，本来在训练中就覆盖了整个圆周，外推并不是大问题。
- **低频维度（高维）**：**强插值**。因为它们负责全局位置，旋转得慢，如果不压缩，稍微走远一点就OOD了。（**为什么容易 OOD？** 由于其波长极长（旋转极慢），在训练的 Context Window 内，这些维度往往连“半圈”都没转完（例如只覆盖了 0 到 10 度的扇区）。一旦推理长度超过训练长度，旋转角度就会进入模型从未见过的数值区域（例如 20 度），导致严重的分布外（OOD）问题。）

**实现技巧：Base Change** 我们不再直接除以 $s$，而是通过改变RoPE的基频参数 $\text{base}$（通常是10000）来实现这种非线性缩放。 令新的基频为 $b' = b \cdot s^{\frac{d}{d-2}}$。 这种变换在数学上等效于：随着维度 $d$ 的增加，缩放因子从1逐渐平滑过渡到 $s$。 结果是：低维度几乎不被压缩（保持精度），高维度被强力压缩（保证视野）。这使得模型在不微调的情况下（Zero-shot）就能获得比线性PI好得多的PPL，微调后更是如虎添翼 。

### 6.3 动态NTK（Dynamic NTK）

静态的NTK缩放有一个缺点：它固定了最大长度。如果你只输入一段很短的文本，它依然应用了缩放，导致性能略微受损。

**动态NTK**在推理时，根据当前输入的实际序列长度 $L_{current}$ 动态计算缩放倍数 $s = \max(1, L_{current} / L_{train})$。

- 当 $L_{current} \le L_{train}$ 时，不缩放，模型表现如初。
- 当 $L_{current}$ 逐渐增加时，缩放力度平滑介入。 这种方法让模型在全长度范围内都能保持最佳状态 。

------

## 7. YaRN (Yet another RoPE extensioN)：集大成者

尽管NTK-Aware Scaling取得了巨大成功，但它在处理超长上下文（如100k+）时仍存在理论缺陷。YaRN的提出，旨在修正这些缺陷，成为目前最完善的RoPE扩展框架。它结合了**NTK-by-parts（分段NTK）**和**熵/温度调节（Temperature Scaling）** 。

### 7.1 NTK-by-parts：精细化的频域手术

YaRN的作者指出，简单的NTK Base Change虽然实现了非线性，但还不够精准。他们提出将维度明确划分为三个频段，使用**斜坡函数（Ramp Function）**进行混合：

1. **高频段（High Frequency）**：波长 $\lambda \ll L_{train}$。
   - 策略：**完全不插值（Extrapolation）**。保持原始旋转速度。
   - 理由：这些维度的旋转周期很短，在训练长度内已经转了无数圈，模型对各个相位的相对关系已经学得很好了，直接外推没问题。
2. **低频段（Low Frequency）**：波长 $\lambda \gg L_{train}$。
   - 策略：**线性插值（Interpolation）**。
   - 理由：这些维度在训练时只转了一点点，外推极其危险，必须压缩回已知范围。
3. **中频段**：过渡区域。

**Ramp Function 定义** ： 定义比率 $r = \frac{L_{train}}{\lambda_d}$。 设置阈值 $\alpha, \beta$。

$$\gamma(d) = \text{clamp}\left(\frac{r - \alpha}{\beta - \alpha}, 0, 1\right)$$

最终的频率 $h(\theta_d)$ 是原始频率和插值频率的加权混合：

$$h(\theta_d) = (1 - \gamma(d)) \frac{\theta_d}{s} + \gamma(d)\theta_d$$

这种分段策略彻底解决了“既要又要”的矛盾：既要保留局部的高精度（高频不插值），又要获得全局的长视野（低频插值）。

### 7.2 熵理论与温度缩放（Temperature Scaling）

这是YaRN最深刻的理论贡献。

研究发现，当我们通过插值扩展上下文时，Attention Mechanism的**分布熵（Distribution Entropy）**会发生变化。 简单来说，当序列变长，$K$ 的数量增加，点积 $QK^T$ 的分布范围会扩大（或因为距离衰减而变得平坦）。这导致Softmax后的概率分布变得比训练时更“尖锐”或更“平坦”，破坏了模型原本的注意力机制——这被称为**分布漂移（Distribution Shift）**。

模型会感到“困惑”：它不知道该聚焦于某一个词，还是平均关注所有词。这直接导致了**“Lost in the Middle”**现象（长文本中间的信息检索不到）。

**解决方案：热力学修正**

YaRN引入了一个温度系数 $t$（Temperature）来修正点积的模长。

$$\text{Attention}(Q, K) = \text{Softmax}\left(\frac{QK^T}{t\sqrt{d_k}}\right)$$

或者在代码实现中，直接将Embedding乘以 $\sqrt{1/t}$。

**$t$ 的计算公式** ： 经过大量实验，作者拟合出了一个经验公式：

$$\sqrt{\frac{1}{t}} = 0.1 \ln(s) + 1$$

其中 $s$ 是扩展倍数。

例如，当扩展倍数 $s=1$ 时，$\sqrt{1/t}=1$，无变化。当 $s$ 很大时，分母变大，相当于降低了Attention Logits的数值，增加了Softmax的熵（使分布更平滑），强迫模型分散注意力，从而覆盖更长的上下文。

### 7.3 YaRN的实战效果

YaRN不仅在PPL指标上优于PI和普通NTK，更重要的是在**Passkey Retrieval（密钥检索）**任务上表现出了惊人的稳健性。它使得Llama 2等模型仅需使用0.1%的微调数据，就能将上下文从4k完美扩展到128k，且几乎没有短文本性能的衰减。

## 8. 现代LLM的演进：2024-2026视角的工程实践

随着Llama 3、Mistral、DeepSeek-V2/V3等模型的发布，RoPE及其变体已成为行业标准，并衍生出更多工程化变体。

### 8.1 Llama 3的豪赌：高Theta策略

在Llama 3中，Meta做出了一个令人惊讶的改动：将RoPE的基频参数 $\text{base}$ 从标准的10,000猛增至**500,000** 。 **深度解读**： 提高 $\text{base}$ 的效果是**拉长了低频维度的波长**。 根据NTK理论，这意味着Llama 3在预训练阶段（Pre-training）就在强迫模型学习超长距离的依赖关系。虽然Llama 3的基础窗口只有8k，但这种高Theta设定为其后续通过微调扩展到1M甚至更长提供了极好的数学基础。它使得长距离的衰减曲线极其平缓，模型拥有了更强的“远视眼”潜力。

### 8.2 Mistral的滑动窗口注意力（SWA）

Mistral 7B采用了一种务实的策略：**Sliding Window Attention (SWA)** 。 RoPE虽然解决了位置编码，但Attention本身的计算复杂度是 $O(L^2)$。对于超长文本，显存和计算量都无法承受。 SWA规定每一层只关注最近的 $W$ 个Token（例如4096）。 **RoPE在SWA中的作用**： 尽管每层只看4k，但RoPE提供的相对位置信息是全局一致的。更妙的是，通过Transformer的多层堆叠，顶层Token的**有效感受野（Receptive Field）**会像CNN一样线性增长（$L \times W$）。这使得Mistral可以用极小的显存开销，处理远超窗口大小的文本逻辑，而RoPE确保了在这个过程中位置关系的精确传递。

### 8.3 多模态的统一：M-RoPE

在Qwen2-VL和PaliGemma等多模态模型中，位置编码面临从1D（文本）向2D（图像）、3D（视频）扩展的挑战。 **M-RoPE (Multimodal RoPE)** 提出了一种极其优雅的方案 ： 利用RoPE维度分块独立的特性，将Embedding向量切分为三部分：

1. **时间部分（Temporal）**：编码视频的帧ID（$t$）。
2. **高度部分（Height）**：编码图像的行ID（$h$）。
3. **宽度部分（Width）**：编码图像的列ID（$w$）。

例如，对于一个视频Patch，其Embedding的前 $d/3$ 维度旋转 $t\theta$，中间 $d/3$ 旋转 $h\theta$，最后 $d/3$ 旋转 $w\theta$。

这种设计使得模型能够同时理解“这是视频的第几秒”、“在画面的哪个角落”，实现了时空位置的统一建模。

### 8.4 DeepSeek-V2的解耦RoPE

DeepSeek-V2引入了MLA（Multi-Head Latent Attention）以极大地压缩KV Cache。然而，MLA采用了低秩压缩（Low-Rank Compression），这导致RoPE无法直接应用于压缩后的Latent Vector（因为旋转会破坏压缩后的语义空间）。 DeepSeek的解决方案是**解耦RoPE（Decoupled RoPE）** ： 将Query和Key向量显式地拆分为两部分：

- **Content Vector**：负责语义，参与压缩，不加RoPE。

- **RoPE Vector**：负责位置，不参与压缩，直接应用RoPE。

  在计算Attention时，将两部分的Score相加。这一设计证明了RoPE作为一种独立的几何位置模块，具有极强的架构兼容性。

------

## 9. 结论与展望

从最初的正弦绝对位置编码，到RoPE的旋转复数域变革，再到YaRN通过NTK理论与热力学熵修正实现的极致长文本扩展，Embedding与位置编码的演进史，就是一部人类试图教会神经网络理解“时空几何”的历史。

回顾这一历程，我们得出以下核心洞察：

1. **几何优于代数**：RoPE的成功证明了，在语义向量空间中，通过**旋转**来表示位置变化，比通过**加法**平移更符合点积注意力的几何直观。它让相对位置不再是需要学习的特征，而是数学上必然涌现的性质。
2. **频域视角的胜利**：NTK-Aware Scaling和YaRN的出现，标志着我们对LLM的理解从单纯的矩阵运算上升到了**信号处理与频域分析**的高度。理解神经网络对不同频率信号的学习偏差，是解决外推问题的关键钥匙。
3. **熵的平衡艺术**：YaRN中的温度缩放提醒我们，长上下文不仅仅是显存和计算量的问题，更是**信息密度与注意力分布**的热力学平衡问题。

展望未来，随着Context Window迈向无限（Infinite Context），位置编码可能会迎来新的范式转移。也许未来的模型将不再依赖固定的RoPE频率，而是能够根据内容自适应地调节“时钟”的转速；又或许，随着线性Attention（Linear Attention）和状态空间模型（SSM, 如Mamba）的复兴，位置编码将与状态方程更加紧密地融合。

但无论形式如何变化，RoPE所确立的“旋转即位置”的几何范式，以及YaRN所强调的“频域控制”思想，都已成为现代大语言模型基因中不可磨灭的一部分。

------

**表格索引：主要位置编码方案对比**

| **方案**             | **核心机制**            | **相对位置实现** | **外推能力** | **优点**                       | **缺点**                               |
| -------------------- | ----------------------- | ---------------- | ------------ | ------------------------------ | -------------------------------------- |
| **APE (Sinusoidal)** | Embedding += Sin(pos)   | 弱 (线性变换)    | 差           | 无需训练                       | 语义污染，无法外推                     |
| **APE (Learned)**    | Embedding += Param[pos] | 弱               | 无           | 灵活适配数据                   | 长度锁死，参数量随长度增加             |
| **ALiBi**            | Score -= m \|i-j\|      | 强 (Bias)        | 强           | 极佳外推，无需Embedding操作    | 表达能力受限，不仅容易兼容KV Cache优化 |
| **RoPE**             | Vector *= Rot(pos)      | 强 (Rotation)    | 中 (需NTK)   | 语义正交，兼容性好，理论优美   | 直接外推PPL爆炸                        |
| **RoPE + PI**        | Rot(pos/s)              | 强               | 好           | 解决PPL爆炸                    | 损失高频分辨率，短文性能降             |
| **NTK-Aware**        | Base Change             | 强               | 优           | 保持高频分辨率，Zero-shot外推  | 极长文仍有瓶颈                         |
| **YaRN**             | NTK-by-parts + Temp     | 强               | 极优         | 完美平衡插值与外推，修正熵漂移 | 实现稍复杂                             |

数据在拥有了位置感之后，即将涌入深层网络进行复杂的特征提取。但在进入注意力机制的‘洪流’之前，还需要一道关键的‘阀门’来确保流动的稳定。 **在下一篇文章中，我们将剖析 Pre-Norm RMSNorm 的设计哲学**。

## 10. 动手实践：Minimind注释版

以下是https://github.com/jingyaogong/minimind的model/model_minimind.py中的位置编码部分，我进行了详细注释

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率矩阵
    
    RoPE 通过旋转矩阵将位置信息编码到 Query 和 Key 中，使模型能够理解 token 的相对位置。
    本函数预计算所有位置的 cos 和 sin 值，避免在每次前向传播时重复计算。
    
    支持 YaRN (Yet another RoPE extensioN) 外推方法，可以处理超过训练时最大长度的序列。
    
    Args:
        dim: 每个注意力头的维度（head_dim）
        end: 最大序列长度（默认 32768）
        rope_base: RoPE 的基频率参数（默认 1e6）
        rope_scaling: RoPE 外推配置字典（YaRN 方法），如果为 None 则不使用外推
        
    Returns:
        freqs_cos: 预计算的 cos 值 [end, dim]
        freqs_sin: 预计算的 sin 值 [end, dim]
    """
    # ========== 步骤 1：计算基础频率 ==========
    # RoPE 频率公式：f_i = 1 / (rope_base^(2i/dim))
    #   其中 i 是维度索引（0, 2, 4, ..., dim-2），只使用偶数索引
    #   频率随维度索引增加而递减，形成不同频率的旋转
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    
    # ========== 步骤 2：应用 YaRN 外推（如果启用） ==========
    if rope_scaling is not None:
        # 获取 YaRN 配置参数
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)  # 训练时的最大长度
        factor = rope_scaling.get("factor", 16)  # 外推因子
        beta_fast = rope_scaling.get("beta_fast", 32.0)  # 快速频率调整参数
        beta_slow = rope_scaling.get("beta_slow", 1.0)  # 慢速频率调整参数
        attn_factor = rope_scaling.get("attention_factor", 1.0)  # 注意力缩放因子
        
        # 如果目标长度超过训练长度，应用 YaRN 外推
        if end / orig_max > 1.0:
            # YaRN 公式：f'(i) = f(i) * ((1-γ) + γ/s)
            #   其中 γ 是线性斜坡函数，s 是缩放因子（factor）
            #   对于低频维度（i < low），不进行缩放
            #   对于高频维度（i > high），完全缩放
            #   对于中间维度，线性插值
            
            # 计算频率调整的边界维度
            # inv_dim(b) 返回频率为 b 的维度索引
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)  # 低频边界
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # 高频边界
            
            # 计算线性斜坡函数 γ
            #   对于维度 i：γ(i) = (i - low) / (high - low)，限制在 [0, 1]
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )
            
            # 应用 YaRN 缩放：f'(i) = f(i) * ((1-γ) + γ/s)
            freqs = freqs * (1 - ramp + ramp / factor)
    
    # ========== 步骤 3：计算所有位置的频率 ==========
    # 为每个位置计算频率：freqs[pos, dim] = pos * freqs[dim]
    t = torch.arange(end, device=freqs.device)  # 位置索引 [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs).float()  # 外积：[end, dim//2]
    
    # ========== 步骤 4：计算 cos 和 sin 值 ==========
    # 将频率转换为 cos 和 sin 值，用于旋转矩阵
    # 由于 RoPE 使用复数旋转，需要将 dim//2 的频率复制到完整的 dim 维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # [end, dim]
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码（RoPE）到 Query 和 Key
    
    RoPE 通过复数旋转将位置信息编码到 Q 和 K 中：
        R_θ(x) = [x_0*cos(θ) - x_1*sin(θ), x_0*sin(θ) + x_1*cos(θ)]
    
    在实现中，将复数旋转分解为实部和虚部的线性组合，使用 rotate_half 函数实现。
    
    Args:
        q: Query 张量 [batch, seq_len, num_heads, head_dim]
        k: Key 张量 [batch, seq_len, num_kv_heads, head_dim]
        cos: 预计算的 cos 值 [seq_len, head_dim]
        sin: 预计算的 sin 值 [seq_len, head_dim]
        position_ids: 位置索引（未使用，cos/sin 已包含位置信息）
        unsqueeze_dim: 在哪个维度插入新维度以匹配 q/k 的形状（默认 1）
        
    Returns:
        q_embed: 应用 RoPE 后的 Query [batch, seq_len, num_heads, head_dim]
        k_embed: 应用 RoPE 后的 Key [batch, seq_len, num_kv_heads, head_dim]
    """
    def rotate_half(x):
        """
        旋转向量的后半部分
        
        将向量分成两半，交换位置并取反后半部分：
            [a, b, c, d] -> [-c, -d, a, b]
        
        这实现了复数旋转的实部/虚部交换。
        
        Args:
            x: 输入张量，最后一个维度会被分成两半
            
        Returns:
            旋转后的张量，形状与输入相同
        """
        # 将最后一个维度分成两半，交换位置并取反后半部分
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    
    # 应用 RoPE 旋转
    # 公式：R_θ(x) = x * cos(θ) + rotate_half(x) * sin(θ)
    #   这等价于复数旋转：x * e^(iθ) = x * (cos(θ) + i*sin(θ))
    #   其中 rotate_half 实现了虚部的操作
    
    # 调整 cos 和 sin 的形状以匹配 q/k：[seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # 对 Query 和 Key 分别应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

