# 基石：**关于Tokenizer你所需要知道的一切**

写给读者：初学者请着重看1,2,3,6章节

## **1. 引言：连接人类认知与机器智能的桥梁**

**为什么电脑能读懂你写的字？**

在当今人工智能的宏大叙事中，大语言模型（Large Language Models, LLMs）如GPT-4、Claude 3和Llama 3等，以其惊人的生成能力和逻辑推理能力占据了舞台的中心。然而，在这些参数量高达数千亿的神经网络开始处理任何信息之前，必须先经过一道至关重要的、却往往被忽视的工序——**分词（Tokenization）**。分词器（Tokenizer）是连接人类自然语言与机器二进制世界的第一道关卡，它将连续的文本流转化为离散的数字序列（Token IDs），使得计算机能够进行计算、统计和理解。

尽管在表面上，分词似乎只是一个简单的字符串预处理步骤，但其背后的数学原理、算法选择以及工程实现细节，对最终模型的性能、推理成本、多语言支持能力甚至安全性都有着深远的影响。分词策略的微小差异，可能导致模型在处理Python代码时缩进混乱，在进行数学运算时位数错误，或者对某些非拉丁语系语言表现出极高的“Token通胀率”  。

我将对Tokenizer进行穷尽式的剖析。我们将从最基础的文本表示概念讲起，深入探讨Byte-Pair Encoding (BPE)、WordPiece和Unigram三种主流算法的数学原理与异同；我们将以一份标准的BPE代码实现为蓝本，逐行解析其Python代码逻辑，揭示训练（Training）与编码（Encoding）的微观运作机制；我们还将深入对比GPT-2与GPT-4分词器的演进，特别是其正则表达式（Regex）预分词模式的工程智慧。

## **2. 文本表示的演变：从字符到子词的必然选择**

在深入代码和具体算法之前，我们必须理解为什么自然语言处理（NLP）领域最终汇聚于“子词（Subword）”分词这一路线上。这是计算效率、词汇覆盖率与语义表达能力之间漫长博弈的结果。

### **2.1 词级别（Word-level）分词的局限**

早期的NLP系统，如基于统计机器翻译（SMT）的模型，往往采用**词级别分词**。这种方法最符合人类的直觉：以空格或标点符号为界，将句子切分为单词。

* **机制**：将 "I love AI." 切分为 ["I", "love", "AI", "."]。  
* **优点**：保留了词的语义完整性，每个Token都对应一个具有明确定义的人类语言概念。  
* **缺陷：词表爆炸（Vocabulary Explosion）**：人类语言具有极其丰富的形态变化。以英语为例，动词 "run" 有 "runs", "running", "ran" 等多种变形；在形态丰富的语言（如土耳其语、芬兰语）中，一个词根通过黏着语素可以生成成千上万种变体。如果将每种变形都视为独立Token，模型需要维护一个数百万量级的巨大词表，这在计算资源上是不可接受的。  
* **未登录词（OOV）问题**：无论词表多大，总会有未见过的词（如新造词 "uninstagrammable" 或人名）。在词级别模型中，这些词只能被映射为通用的 \<UNK>（Unknown）标记，导致该位置的信息完全丢失，这对翻译或理解任务是灾难性的  。

### **2.2 字符级别（Character-level）分词的尝试**

为了彻底解决OOV问题，研究者一度转向**字符级别分词**，将文本拆解为单个字符。

* **机制**：将 "love" 切分为 ["l", "o", "v", "e"]。  
* **优点**：词表极小（仅需包含字母表、数字和符号，通常在100-1000量级），理论上消灭了OOV问题。  
* **缺陷：序列过长与语义稀疏**：  
  1. **计算成本**：Transformer模型的注意力机制（Self-Attention）的时间复杂度与序列长度的平方成正比（$O(N^2)$）。字符级分词会导致序列长度增加5-10倍，使得训练和推理成本指数级上升。  
  2. **语义缺失**：单个字符（如 "t"）通常不承载独立的语义信息。模型需要花费大量的层数和参数去组合字符以识别出“词”的概念，这浪费了模型的表达能力。

### **2.3 子词（Subword）分词的崛起**

子词分词（Subword Tokenization）是上述两种方法的辩证统一，也是目前大语言模型的标准配置。其核心哲学是：**常用词保持完整，罕见词拆分为有意义的子部件（Subword units）**。

例如，单词 “tokenization” 在子词分词器中可能被拆分为 “token” 和 “ization”。

* “token” 是高频词根，作为一个整体被保留，模型可以直接获取其语义嵌入。  
* “ization” 是常见的高频后缀，作为一个整体被保留。  
* 这种机制使得模型既能高效处理常见词，又能通过组合词根和词缀来泛化理解未见过的复合词（如 “modernization”, “optimization”），从而在有限的词表大小下实现了无限的词汇表达能力。

目前主流的子词算法三巨头包括：

1. **BPE (Byte-Pair Encoding)**：基于频率的合并策略，广泛用于GPT系列、Llama、RoBERTa。  
2. **WordPiece**：基于概率（似然度）的合并策略，起源于BERT。  
3. **Unigram**：基于概率的剪枝策略（从大词表删减），用于SentencePiece (ALBert, T5)。

## **3. 深度解析：Byte-Pair Encoding (BPE) 算法与代码实现**

Byte-Pair Encoding（字节对编码）最初是作为一种数据压缩算法由Philip Gage在1994年提出的。Sennrich等人于2015年将其引入NLP领域，用于解决神经机器翻译中的稀有词问题。如今，它已成为GPT系列模型的基石。

为了彻底理解BPE，我们将不仅仅停留在理论层面，而是通过**逐行解析代码**的方式，通过一个参考实现（基于Karpathy的 minbpe 逻辑）来剖析其内部运作。

### **3.1 算法核心逻辑与代码结构**

BPE的训练过程本质上是一个迭代的**数据压缩**过程。

1. **初始化**：将所有文本拆解为基础单元（通常是字节）。  
2. **统计**：统计所有相邻单元对（Pair）在数据中出现的频率。  
3. **合并**：找到频率最高的那个对（例如 ('e', 's')），将其合并为一个新的符号（'es'），并分配一个新的ID。  
4. **迭代**：重复步骤2和3，直到达到预设的词表大小（Vocabulary Size）。

我们将代码分为三个核心功能模块进行讲解：**统计（Statistics）**、**合并（Merge）** 和 **训练主循环（Training Loop）**。

### **3.2 代码详解：统计频率 (get_stats)**

这是BPE算法中最基础的原子操作：扫描当前的Token序列，统计每一对相邻Token出现的次数。

```python
def get_stats(ids):  
    """  
    输入:  
    ids (list of integers): 当前的Token ID列表。  
    输出:  
    counts (dict): 映射 (id1, id2) -> frequency  
    """  
    counts = {}  
    # zip(ids, ids[1:]) 是一个Python技巧    
    # zip后生成: [(1, 2), (2, 3), (3, 4)]  
    # 这正是我们需要统计的所有相邻对  
    for pair in zip(ids, ids[1:]):   
        counts[pair] = counts.get(pair, 0) + 1  
    return counts
```

**深入解析与知识点：**

* **输入数据的本质**：这里的 ids 最初是什么？在现代LLM（如GPT-4）中，我们使用**字节级（Byte-level）BPE**。这意味着 ids 的初始状态是文本UTF-8编码后的字节序列，范围在0-255之间。例如，英文字母 a 是97，中文 你 是三个字节。这种设计极其重要，因为它保证了Tokenizer可以处理任何Unicode字符串，哪怕是Emoji或从未见过的外文，因为一切皆为字节。  
* **时间复杂度**：此函数的时间复杂度为 $O(N)$，其中 $N$ 是序列长度。在训练过程中，每次合并都需要重新扫描整个语料库，这使得朴素实现的BPE训练非常慢（$O(N \cdot V)$，其中 $V$ 是合并次数）。工业级实现（如Rust编写的Tiktoken）会使用链表或优先队列来优化更新过程，避免全量扫描。

### **3.3 代码详解：执行合并 (merge)**

一旦我们找到了出现频率最高的对（比如 (101, 115) 对应 ('e', 's')），我们就需要将序列中所有的 (101, 115) 替换为新的Token ID（比如 256）。

```python
def merge(ids, pair, idx):  
    """  
    输入:  
    ids (list): 当前的Token列表  
    pair (tuple): 要合并的一对Token, 例如 (101, 115)  
    idx (int): 分配给这个新Token的ID, 例如 256  
      

    输出:  
    newids (list): 合并后的新列表  
    """  
    newids = []  
    i = 0  
    while i < len(ids):  
        #检查是否刚好碰到了我们要合并的pair，且没有越界  
        #注意: i < len(ids) - 1 是为了防止检查 ids[i+1] 时越界  
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:  
            newids.append(idx) # 替换为新的ID  
            i += 2 # 跳过接下来两个元素，因为它们已经被合并了  
        else:  
            newids.append(ids[i]) # 保持原样  
            i += 1  
    return newids
```

**深入解析与知识点：**

* **贪婪算法（Greedy Strategy）**：BPE是贪婪的。只要选定了频率最高的Pair，它就会在**全局**范围内将所有的实例都替换掉。这不同于某些动态规划算法。  
* **序列变短**：每次合并操作都会使 ids 列表的长度变短。这正是“压缩”的体现。对于大模型，更短的序列意味着能放入更多的上下文信息。  
* **新ID的分配**：基础词表是0-255。第一次合并产生的Token ID是256，第二次是257，依此类推。GPT-4的 cl100k_base 词表大小约为100,277，这意味着这个合并过程在训练阶段重复了约10万次。

### **3.4 代码详解：训练主循环 (train)**

将上述两个函数结合，就构成了BPE的训练器。

```python
def train(text, vocab_size, verbose=False):  
    """  
    输入:  
    text (str): 训练语料文本  
    vocab_size (int): 目标词表大小 (例如 50257)  
    """  
    assert vocab_size >= 256  
    num_merges = vocab_size - 256 # 需要进行的合并次数  
      

    # 1. 预处理：将文本转换为UTF-8字节流  
    text_bytes = text.encode("utf-8")  
    ids = list(text_bytes) # 初始列表，元素范围 0-255  
      
    merges = {} # 记录合并规则: (p0, p1) -> idx  
      
    print(f"Original length: {len(ids)}")  
      
    for i in range(num_merges):  
        # 2. 统计当前频率  
        stats = get_stats(ids)  
        if not stats:  
            break # 如果没有可以合并的对，提前退出  
              
        # 3. 找到频率最高的对  
        # key=stats.get 表示按字典的值(频率)排序  
        pair = max(stats, key=stats.get)  
          
        # 4. 分配新ID  
        idx = 256 + i  
          
        # 5. 执行合并  
        ids = merge(ids, pair, idx)  
          
        # 6. 记录规则  
        merges[pair] = idx  
          
        if verbose:  
            print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} (count: {stats[pair]})")  
              
    return merges
```

**深入解析与知识点：**

* **核心产物**：训练结束后，最重要的产物不是 ids，而是 merges 字典。这个字典定义了Tokenizer的“知识”。当我们下载一个预训练模型时，tokenizer.json 或 merges.txt 存储的就是这个字典。  
* **确定性与Tie-breaking**：如果两个Pair频率相同怎么办？Python的 max 函数在值相同时会返回第一个遇到的键。为了保证Tokenizer的可复现性（Deterministic），在工业级实现中通常会规定：当频率相同时，按照Pair中字符的字典序（Lexicographical order）进行选择。  
* **词表大小的权衡**：vocab_size 是一个关键超参数。  
  * **太小**：导致序列过长，模型推理慢，无法捕捉长距离依赖。  
  * **太大**：导致Embedding矩阵（vocab_size x hidden_dim）参数量激增，增加训练负担；且由于稀有词频次太低，其Embedding可能训练不充分（Undertrained）。

### **3.5 代码详解：推理阶段的编码 (encode)**

有了训练好的 merges 规则，如何将新的文本转化为Token IDs？这是一个极易出错的环节。初学者常犯的错误是：在推理时依然每次找文本中频率最高的对。**这是错误的**。推理时，必须严格按照**训练时确定的优先级顺序**进行合并。

```python
def encode(text, merges):  
    # 1. 转为字节流  
    ids = list(text.encode("utf-8"))  
      

    while len(ids) >= 2:  
        # 获取当前文本中所有相邻对  
        stats = get_stats(ids)  
          
        # 寻找“在merges规则表中存在，且ID最小（即最早被训练出来）”的对  
        # 因为ID越小，说明它在训练集中频率越高，优先级越高  
        pair_to_merge = None  
        min_rank = float("inf") # rank即ID  
          
        for pair in stats:  
            if pair in merges:  
                rank = merges[pair]  
                if rank < min_rank:  
                    min_rank = rank  
                    pair_to_merge = pair  
          
        # 如果当前序列中没有任何对在我们的规则表中，停止  
        if pair_to_merge is None:  
            break  
              
        # 执行合并  
        ids = merge(ids, pair_to_merge, min_rank)  
          
    return ids
```

**深入解析与知识点：**

* **优先级逻辑**：代码中的 min_rank 逻辑至关重要。假设我们有规则 (a, b) -> X (Rank 1) 和 (b, c) -> Y (Rank 2)。对于输入 abc，我们必须先合并Rank 1的规则，得到 Xc。如果我们先合并 bc 得到 aY，就破坏了BPE的构造一致性，导致生成的Token序列与训练时分布不一致。  
* **递归与迭代**：上述代码使用了类似迭代的方式。在Python的 minbpe 实现中，也可以通过递归来实现，或者使用更高效的Regex拆分后独立处理每个块（详见后文GPT-2/4部分）。

### **3.6 代码详解：解码 (decode)**

解码是编码的逆过程，相对简单，但有一个关键细节：处理无效字节。

```python
def decode(ids, vocab):  
    """  
    ids: token ID列表  
    vocab: 映射 {idx: bytes} (由基础字节和merges反推得到)  
    """  
    # 将所有ID映射回字节串，并拼接  
    tokens = b"".join(vocab[idx] for idx in ids)  
      

    # errors="replace" 是关键  
    text = tokens.decode("utf-8", errors="replace")  
    return text
```

**深入解析与知识点：**

* **errors="replace"**：在GPT-3或GPT-4的输出中，我们偶尔会看到 (Replacement Character)。这是因为大模型有时会生成不完整的Token序列。例如，一个中文字符由3个字节组成，如果模型只输出了前2个字节就停止了（比如因为`max_tokens`限制），那么这2个字节无法构成合法的UTF-8字符。使用 `replace` 策略可以防止程序崩溃。

## **4. 算法三巨头深度对比：BPE, WordPiece 与 Unigram**

虽然BPE占据了GPT系列的主导地位，但在BERT、T5等模型中，WordPiece和Unigram算法同样重要。理解它们的区别是深入NLP底层逻辑的关键。

### **4.1 WordPiece：从频率到概率的跨越**

WordPiece算法由Google开发，是BERT模型的核心。它的整体流程与BPE非常相似（也是自底向上的合并），但在**选择合并哪一对**的标准上完全不同。

* **BPE标准**：选择**频次（Frequency）**最高的对。  
  * 目标：最大化数据压缩比。  
* **WordPiece标准**：选择合并后能使训练数据**似然度（Likelihood）**增加最多的对。  
  
  * 这等价于选择**点互信息（Pointwise Mutual Information, PMI）**最高的对。  
  * WordPiece 得分公式：
  
    $$text{Score}(A, B) = \frac{P(AB)}{P(A) \times P(B)}$$
  
    其中 $P(AB)$ 是对 $AB$ 出现的概率，$P(A)$ 和 $P(B)$ 是各自的概率。

**深度洞察：为什么PMI比频率更优？** 
WordPiece的评分机制考虑了子词的独立概率。

* 假设 A="the" 和 B="book" 都是极高频词，那么它们连在一起 thebook 出现的次数可能很高。但在BPE中它们可能会被合并。  
* 在WordPiece中，由于分母 $P(\text{"the"}) \times P(\text{"book"})$ 非常大，Score 会被拉低。这防止了两个本该独立的常见词被意外合并。  
* 相反，假设 A="Z" 和 B="qa" 都很罕见，但只要出现就总是粘在一起（如 "Zqa"）。此时 $P(AB) \approx P(A) \approx P(B)$，Score 会非常高（接近 $1/P(A)$）。  
* **结论**：WordPiece倾向于合并那些**内在关联性强**（比随机组合更紧密）的词对，而不仅仅是高频词对。这使得WordPiece在处理词缀（如 "un-", "-ing"）时通常比BPE在语言学上更合理。

### **4.2 Unigram：基于概率图模型的自顶向下剪枝**

Unigram Tokenization（主要在SentencePiece库中实现）的思路与前两者截然相反。BPE和WordPiece是**自底向上（Bottom-up）**的构造，Unigram是**自顶向下（Top-down）**的。

数学原理与EM算法： 
Unigram模型假设每个Subword的出现是独立的。一个句子 $X$ 被切分为序列 $\mathbf{x} = (x_1,..., x_m)$ 的概率为：

$$P(\mathbf{x}) = \prod_{i=1}^{m} P(x_i)$$

其中 $P(x_i)$ 是子词 $x_i$ 的发生概率。  

**训练流程（EM算法）：**

1. **初始化**：构建一个极其巨大的词表（例如包含语料中所有出现过的子串，可能有几百万个）。  
2. **E-step（期望步）**：固定当前词表，使用**Viterbi算法**计算语料库中每个句子的最优切分路径。  
   * 对于单词 "tokenization"，可能有多种切分方式（["token", "ization"] vs ["t", "o", "ken", "..."]）。Viterbi算法能找到使得联合概率 $P(\mathbf{x})$ 最大的那条路径。  
3. **M-step（最大化步）**：重新计算每个子词的出现概率 $P(x_i)$。  
4. 计算Loss并剪枝：计算如果从词表中移除某个子词 $x$，总似然度 $L$ 会下降多少（Loss）。

   $$\Delta L = L_{new} - L_{old}$$  
5. **剪枝策略**：移除那些对总似然度贡献最小（Loss最小）的Token（通常每轮移除20%）。  
6. **循环**：重复上述过程直到词表缩小到预定大小。

**独特优势：子词正则化（Subword Regularization）：**


Unigram不仅仅是一个分词器，它本身就是一个微型的语言模型。在训练LLM时，我们可以利用Unigram的概率特性进行数据增强。

* 对于同一个词 "New York"，我们不总是输出最优切分 ["New", " York"]。  
* 我们可以根据概率采样（Sampling），有时将其切分为["N", "ew", " Yo", "rk"]。  
* 这种技术迫使模型学习不同切分下的语义，显著提升了模型在处理拼写错误和噪声文本时的鲁棒性。

### **4.3 算法特性对比总结表**

| 特性 | BPE (GPT-2/3/4, Llama) | WordPiece (BERT) | Unigram (T5, ALBERT) |
| :---- | :---- | :---- | :---- |
| **构建方向** | 自底向上 (Bottom-up) | 自底向上 (Bottom-up) | 自顶向下 (Top-down) |
| **核心指标** | 频率 (Frequency) | 似然度增益 / PMI | 似然度损失 (Loss) |
| **训练复杂度** | 较低 | 高 (每步需重算所有Pair得分) | 较高 (需EM迭代) |
| **OOV处理** | 字节回退 (Byte fallback) | 字符回退 (需UNK Token) | 字符回退 |
| **分词确定性** | 确定性 (Deterministic) | 确定性 | 概率性 (可采样多种结果) |
| **适用场景** | 生成式模型 (Generative) | 理解式模型 (NLU) | 需正则化的任务 |

## **5. GPT分词器的演进：从GPT-2到GPT-4的工程飞跃**

OpenAI的GPT系列一直沿用BPE算法，但在细节上进行了极其关键的优化。了解这些演进，能让我们明白为什么GPT-4在写代码和处理非英语语言时比GPT-2强得多。

### **5.1 预分词（Pre-tokenization）与Regex的秘密**

BPE算法本身是“盲目”的。如果不加干预，它可能会跨越标点和单词的边界进行合并。例如，它可能会把句子结尾的 "dog." 中的 "g." 合并为一个Token。这通常是不希望看到的，因为标点符号通常有独立的语法意义。

因此，在运行BPE合并之前，需要先用**正则表达式（Regex）**将文本切分为一个个基础的“单词块”。BPE只能在这些块内部进行合并，不能跨块。

#### **5.1.1 GPT-2 Regex 模式分析**

GPT-2使用的Regex模式如下（Python re 语法）：

```python
r"""'(?:[sdmt]|ll|ve|re)|?p{L}+|?p{N}+|?[^sp{L}p{N}]+|s+(?!S)|s+"""
```

**逐段解析：**

1. '(?:[sdmt]|ll|ve|re)：**处理缩写**。它将 's, 't, 're 等缩写从单词中剥离出来。例如 "don't" 会被拆分为 "don" 和 "'t"。这保证了模型能理解否定、所有格等概念。  
2. ?p{L}+：**处理单词**。匹配前导空格（可选）加上一串字母。注意，GPT-2将空格视为单词的一部分（通常在开头），如 token。  
3. ?p{N}+：**处理数字**。匹配连续的数字。  
4. ?\[^sp{L}p{N}]+：**处理标点**。匹配非空格、非字母、非数字的字符序列。  
5. s+(?!S)：**处理尾部空格**。

**GPT-2的缺陷：**  

这个正则在处理多空格时表现不佳，且对代码中的缩进（通常是大量空格）处理效率极低。更重要的是，它对大小写的处理不够完美。例如，它能识别 's 为缩写，但如果用户输入大写的 HOW'S，由于正则中没有忽略大小写的标志，'S 可能不会被单独切分，从而导致分词不一致。

#### **5.1.2 GPT-4 (cl100k_base) Regex 模式分析**

GPT-4（以及Tiktoken库）使用了更复杂的Regex：

```python
r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^rnp{L}p{N}]?p{L}+|p{N}{2,}|[^rnp{L}p{N}]?[^sp{L}p{N}]+[rn]*|s*[rn]+|s+(?!S)|s+"""
```

**重大改进与洞察：**

1. **大小写不敏感 (?i:...)**：解决了GPT-2中 Don't 和 DON'T 切分不一致的问题。这看似微小，实则极大提升了模型对大写输入的理解能力。  
2. **数字合并策略 p{N}{2,}**：GPT-4要求数字匹配至少两位。这倾向于将数字进行更激进的切分，或者限制数字合并的长度。这有助于提升数学计算能力，防止非常长的数字串被合成一个极其稀有的Token，导致模型无法理解其数值含义。  
3. **空格与代码优化**：GPT-4的正则允许合并更多的连续空格。这对于**Python代码**（依赖缩进）极其重要。GPT-2往往把4个空格切成4个Token，浪费了宝贵的上下文窗口（Context Window），而GPT-4可以将其压缩为一个Token，大幅提升了处理长代码的能力。

### 5.2 字节级BPE的精妙实现 (bytes_to_unicode)

在 GPT-2 的源码中，`bytes_to_unicode` 是一个极具迷惑性但至关重要的函数。它是 BPE 算法能够处理任意字节流（包括 Unicode 乱码、二进制文件、Emoji 等）的基石。

#### 5.2.1 核心痛点：部分字节是不可见的
BPE 算法运行在 **字节（Byte）** 级别上，这意味着它需要处理 0-255 的所有可能取值。然而，这带来了一个巨大的工程问题：

* **不可见性与干扰：** 很多字节是不可见的控制字符（如空格、换行符 `\n`、制表符 `\t`），甚至是无效的二进制数据（如 `0x00`）。
* **调试灾难：** 如果直接将这些字节作为 Token 处理，打印调试信息时终端会乱码、换行错位甚至发出蜂鸣声。更重要的是，在这个阶段，**人类肉眼无法区分“空字符串”和“空格字符”**。

#### 5.2.2 解决方案：全射映射（Bijective Mapping）
OpenAI 并没有过滤掉这些特殊字节，而是给它们穿上了一层“可视化的外衣”。函数构建了一个 **可逆的映射表**：

1.  **可见字符保持原样：** 标准的 ASCII 可打印字符（如 `a-z`, `A-Z`, `0-9`, `!`, `?`）保持不变，直接映射为对应的 Unicode 字符。
2.  **不可见字符转换：** 将所有不可见字符（空格、控制符）和非 ASCII 字节，映射到 Unicode 编码表中 **256 号之后** 的可见字符区域（通常是拉丁字母扩展区）。

**最经典的映射案例：空格（Space）**
* **原始字节：** 32 (`0x20`) —— 在屏幕上不可见。
* **映射结果：** **`Ġ`** (U+0120, Latin Capital Letter G with Dot Above)。
* **目的：** 让空格在分词结果中变得清晰可见。这就是为什么我们在 GPT 的 Token 列表中看到的总是 `Ġworld` 而不是 ` world`。

#### 5.2.3 数据的生命周期（从输入到输出）

要理解这个实现的精妙，必须认识到这种转换不仅仅是“显示特效”，而是**数据层面的实质改变**。

| 阶段                 | 数据形态            | 说明                                                         |
| :------------------- | :------------------ | :----------------------------------------------------------- |
| **Step 1: 原始输入** | `b'Hi world'`       | 包含原始字节，空格是 `0x20`。                                |
| **Step 2: 映射转换** | `"HiĠworld"`        | **关键一步**。字节 `0x20` 被替换为字符 `Ġ`。此后的 BPE 算法完全基于这个新字符串运行。 |
| **Step 3: 词表存储** | `{"Ġworld": 12345}` | 在 `vocab.json` 中，Token 是以 `Ġ` 的形式真实存储的。模型训练也是针对 `Ġ` 进行的。 |
| **Step 4: 解码还原** | `b'Hi world'`       | **逆向过程**。当模型输出 Token 后，Decoder 会查找反向映射表，将 `Ġ` 还原回字节 `0x20`。 |
| **Step 5: 最终展示** | `"Hi world"`        | 字节流经过 UTF-8 解码，呈现给用户正常的文本。                |

#### 5.2.4 代码意义总结
这个函数实现了三个目标，体现了工程实现的优雅：

1.  **完整性 (Lossless)：** 256 个字节对应 256 个 Unicode 字符，没有丢失任何信息。
2.  **可读性 (Readability)：** 所有的中间状态（分词结果、词表文件）都是人类可读的字符串，极大降低了调试难度。
3.  **通用性 (Universality)：** 这种处理方式使得 GPT-2 不仅能生成文本，理论上也能处理和生成任何二进制数据（如图片字节流），因为模型本质上只是在预测下一个 ID，而不关心这个 ID 代表的是字母还是换行符。

### **5.3 词表大小的演进逻辑**

| 模型 | 词表大小 | 含义 | 影响分析 |
| :---- | :---- | :---- | :---- |
| **GPT-2** | 50,257 | 较小 | 英语为主，其他语言效率低。 |
| **GPT-4** | 100,277 | 翻倍 | 显著提升多语言压缩率，代码效率提升。 |
| **Llama 3** | 128,000 | 更大 | 进一步优化多语言支持。 |

为什么词表越来越大？  

更大的词表意味着同一个句子被切分成更少的Token。

* **优点**：  
  1. **推理更快**：生成的步数变少了。  
  2. **上下文更“大”**：同样的Token限制下能塞进更多的实际文本。  
  3. **多语言公平性**：大词表允许容纳更多非英语语言的常用词（如中文单字），减少非英语文本的“Token膨胀”率。  
* **代价**：Embedding层参数量剧增（128k * 4096 维度的矩阵非常巨大），训练收敛更难，需要更多的数据来填满这些稀疏的Token嵌入。

## **6. 特殊Token (Special Tokens) 的工程处理机制**

在代码实现中，特殊Token（如 <|endoftext|>, \<PAD>, \<MASK>, <|im_start|>) 的处理往往是初学者最容易混淆的地方，也是导致模型产生幻觉或安全漏洞的常见原因。

### **6.1 为什么不能用普通BPE处理特殊Token？**

假设我们的特殊Token是 <|endoftext|>，用于标记文本结束。如果我们把它当作普通文本传给BPE算法：

1. BPE会先将其按字符拆分。  
2. 然后按照合并规则，它可能会被切分为 ['<', '|', 'endo', 'ft', 'ext', '|', '>']。  
3. **后果**：模型将无法把这个序列识别为一个统一的“停止信号”，而是一堆无意义的碎片。模型可能无法正确停止生成。

### **6.2 解决方案：Tiktoken vs HuggingFace**

Tiktoken (GPT-4) 的处理方式：  
Tiktoken 要求用户显式传递 allowed_special 参数。这是为了安全。

* **Prompt注入防御**：如果不允许特殊Token，当用户输入 "Hello <|endoftext|>" 时，Tiktoken会强制将其作为普通文本进行分词（即切碎），防止用户伪造系统指令。  
* **实现机制**：Tiktoken 内部维护了一个独立的字典来存储特殊Token。在分词开始前，它会先用正则把这些特殊字符串“抠”出来，不参与BPE合并，直接赋予特定的ID。

HuggingFace Tokenizers 的处理方式：  
HF 引入了 AddedToken 对象，并区分 special_tokens（具有特殊语义，如EOS）和 additional_special_tokens。HF的分词流程中包含一个 Normalization 步骤，但在处理特殊Token时，它同样会优先保护这些Token不被切分。  
代码实战：如何在自制Tokenizer中处理？  
如果你在使用 minbpe 或自己写 tokenizer，必须在 encode 函数的最开始加入对特殊Token的正则匹配。

```python
# 伪代码逻辑：处理特殊Token  
def encode(text, special_tokens):  
    # special_tokens 是一个字典 {'<|endoftext|>': 100257}  
      

    # 1. 创建一个正则模式，匹配所有特殊Token  
    # 例如 pattern = "(<|endoftext|>)"  
    special_pattern = create_pattern(special_tokens.keys())  
      
    # 2. 将文本切分为“普通部分”和“特殊部分”  
    # text: "Hello <|endoftext|> world"  
    # splits: ["Hello ", "<|endoftext|>", " world"]  
    splits = re.split(special_pattern, text)  
      
    final_ids = []  
    for part in splits:  
        if part in special_tokens:  
            # 如果是特殊Token，直接追加ID  
            final_ids.append(special_tokens[part])  
        else:  
            # 如果是普通文本，应用BPE算法  
            final_ids.extend(bpe_encode(part))  
              
    return final_ids
```

## **7. 分词对模型性能的深远影响**

Tokenizer不仅仅是数据的搬运工，它实际上重塑了模型眼中的世界。许多LLM的怪异行为都可以追溯到分词阶段。

### **7.1 算术与数字的“盲区”**

LLM通常在算术任务上表现挣扎，部分原因在于分词。

* **不一致性**：1000 可能是一个Token。1001 可能被切分为["100", "1"] 。`1002` 可能是["10", "02"] 。  
* **位值丢失**：由于数字被切分得支离破碎且不规律，模型很难学习到统一的“位值（Place Value）”规则（即个位、十位、百位的关系）。  
* **GPT-4的改进**：通过正则限制数字的合并，尽量保持数字切分的一致性，但这依然是基于文本模型的硬伤。

### **7.2 编程语言的缩进处理**

在Python代码中，缩进即逻辑。

* 在GPT-2 Tokenizer中，4个空格通常被切分为4个 Ġ Token。这意味着一段深度缩进的代码会消耗大量的Token配额，且模型必须精确数出有多少个 Ġ 才能确定代码块层级。  
* GPT-4通过将连续空格合并，显著缓解了这个问题，使得模型在生成代码时逻辑更严密，上下文利用率更高。

### **7.3 “Glitch Tokens” (故障Token)**

研究人员发现，某些Token（如 SolidGoldMagikarp, Dragonbound）会导致模型生成乱码或崩溃。

* **原因**：这些词通常来自Reddit等网络论坛的用户名，在爬取数据时被BPE统计为高频词并加入了词表。  
* **灾难**：然而，在后续的训练数据清洗中，这些词可能被当作“噪声”过滤掉了。  
* **结果**：导致这些Token存在于词表中，但在Embedding层的训练中从未被更新过（处于初始随机状态）。当模型在推理时偶尔遇到这些词，就会激活一个未经训练的随机向量，导致输出崩坏。这提醒我们在构建Tokenizer时，数据的预处理和后处理必须严格对齐。

## **8. 性能与生态：Tiktoken, SentencePiece, HuggingFace**

在工程实践中，我们很少直接运行纯Python版的BPE，因为效率太低。我们通常使用高度优化的库。

### **8.1 核心库对比**

| 库 | 核心语言 | 支持算法 | 特点 | 适用模型 |
| :---- | :---- | :---- | :---- | :---- |
| **Tiktoken** (OpenAI) | Rust | BPE | **极速**（比HF快3-6倍）。专为OpenAI模型优化。API简单，仅支持推理，不支持训练新模型（通常）。 | GPT-3.5, GPT-4 |
| **SentencePiece** (Google) | C++ | BPE, Unigram | **无损处理**（Lossless）。将空格视为特殊字符 _，不需要预分词正则。完全可逆。对多语言支持极好。 | Llama, ALBERT, T5 |
| **Tokenizers** (HuggingFace) | Rust | All | **大一统**。功能最全，支持训练和推理。集成了上述两者的优点，但API相对复杂。 | BERT, RoBERTa, Mistral |

### **8.2 SentencePiece 的“空格”处理**

传统的Tokenizer（如BERT）先按空格分词，这就丢失了“这里原来有几个空格”的信息。SentencePiece将空格视为一个普通字符（用下划线 _ 或 (U+2581) 表示）。

* 输入：Hello World (两个空格)  
* SP分词：["\_Hello", "\_", "World"] (保留了所有信息)  
* 这就是为什么Llama等模型可以直接处理原始文本，而不需要复杂的预处理规则。这种“Raw stream in, Token stream out”的设计理念是目前开源大模型的主流。

## **9. 结论与未来展望**

Tokenization 是自然语言处理中一个古老而又充满活力的领域。从早期的空格切分，到统计学的BPE，再到如今融合了语言学规则与概率图模型的复杂系统，Tokenizer 的每一次进化都推动了模型性能的边界。

**未来的方向在哪里？**

1. **Token-free Models (Byte-level Transformers)**：研究者正在探索直接在字节层面（MegaByte, MambaByte）进行训练。虽然序列长度增加了4倍，但随着Flash Attention等线性注意力机制的发展，这正在成为可能。这将彻底消灭Tokenizer带来的所有偏见和问题。  
2. **多模态融合**：随着GPT-4o等模型的出现，Tokenizer不仅要处理文本，还要处理图像Patch和音频帧。未来的Tokenizer将是“万物皆Token”的统一体。

通过理解Tokenizer的每一个字节、每一行代码、每一个正则符号，我们不仅是在学习一个预处理工具，更是在窥探大语言模型认知世界的“第一眼”。这一眼，决定了它能看多远。

## 10. Minimind的tokenizer实现

看完上面的介绍后，你应该可以流畅地看懂Minimind做了什么。我给Minimind的tokenizer代码写了详细注释。

```python
"""
MiniMind 分词器训练脚本

本脚本用于从 JSONL 格式的预训练数据集中训练 BPE（Byte Pair Encoding）分词器，
并配置为兼容 Hugging Face Transformers 的标准格式。

主要功能：
1. 从 JSONL 数据集读取文本数据
2. 使用 BPE 算法训练分词器，词汇表大小为 6400
3. 定义并配置特殊 Token（<|endoftext|>, <|im_start|>, <|im_end|>）
4. 保存分词器为 Hugging Face 兼容格式
5. 配置聊天模板（chat_template）用于多轮对话格式化

输出文件（保存在 ./model/ 目录）：
- tokenizer.json: 分词器核心配置（BPE 模型、预处理/解码逻辑、词汇表）
- tokenizer_config.json: Transformers 兼容配置（聊天模板、特殊 Token 映射等）
- vocab.json: BPE 词汇表
- merges.txt: BPE 合并规则

依赖库：
- tokenizers: Hugging Face 的快速分词器库
- transformers: Hugging Face Transformers 库（用于验证和加载）
"""

import random
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os

from transformers import AutoTokenizer

# 设置随机种子，确保结果可复现
random.seed(42)


def train_tokenizer():
    """
    从 JSONL 数据集训练 BPE 分词器，定义特殊 Token，保存为标准格式。
    
    功能概述：
    1. 从 JSONL 文件读取文本数据
    2. 初始化 BPE 分词器并配置 ByteLevel 预处理
    3. 定义特殊 Token（文本结束符、消息开始/结束符）
    4. 使用 BPE 算法训练分词器，构建词汇表
    5. 验证特殊 Token 的 ID 顺序
    6. 保存分词器为 Hugging Face 兼容格式
    7. 创建 tokenizer_config.json 配置文件，包含聊天模板
    
    输出：
        - 在 ./model/ 目录下生成分词器相关文件
        - tokenizer.json: 核心分词器配置
        - tokenizer_config.json: Transformers 兼容配置
        - vocab.json 和 merges.txt: BPE 词汇表和合并规则
    
    注意：
        - 特殊 Token 的 ID 必须严格按照顺序：endoftext=0, im_start=1, im_end=2
        - 聊天模板依赖这些 ID，顺序错误会导致格式化失败
    """
    # ========== 步骤 1：读取 JSONL 数据集 ==========
    def read_texts_from_jsonl(file_path):
        """
        从 JSONL 文件中逐行读取文本数据。
        
        JSONL 格式：每行是一个 JSON 对象，包含 'text' 字段
        例如：{"text": "这是一段训练文本"}
        
        Args:
            file_path: JSONL 文件路径
            
        Yields:
            str: 每行 JSON 对象中的 'text' 字段内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    
    # 预训练数据路径（JSONL 格式，每行包含 {"text": "..."}）
    data_path = '/home/zmm/minimind/dataset/pretrain_hq.jsonl'

    # ========== 步骤 2：初始化 BPE 分词器 ==========
    # BPE（Byte Pair Encoding）算法原理：
    #   1. 从单字节（256 个可能的字节值）开始
    #   2. 迭代地找到并合并出现频率最高的字符对（byte pair）
    #   3. 重复合并过程，直到达到目标词汇表大小
    #   优点：能平衡词汇量大小和编码效率，既不会过于冗余，也不会丢失信息
    #
    # ByteLevel 预处理机制：
    #   - 先将文本按 UTF-8 编码拆分为字节序列
    #   - 例如："你好" → UTF-8 字节序列 [0xe4, 0xbd, 0xa0, 0xe5, 0xa5, 0xbd]
    #   - 然后再对这些字节进行 BPE 合并
    #   优点：避免 OOV（Out-Of-Vocabulary，未登录词）问题，因为任何文本都可以表示为字节序列
    tokenizer = Tokenizer(models.BPE())
    # add_prefix_space=False: 不在文本开头添加空格
    #   设置为 False 是因为中文等语言不需要在开头添加空格
    #   如果设置为 True，会在每个文本前添加空格（适合英文等需要空格分隔的语言）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ========== 步骤 3：定义特殊 Token ==========
    # 特殊 Token 的作用：
    #   1. <|endoftext|> (ID=0): 
    #      - 文本结束符（End of Text）
    #      - 同时用作填充符（pad_token）和未知词（unk_token）
    #      - 在序列末尾或需要填充时使用
    #   2. <|im_start|> (ID=1):
    #      - 聊天消息开始符（Begin of Message）
    #      - 标记每条消息的开始，格式：<|im_start|>role\ncontent
    #      - 也用作序列开始符（bos_token）
    #   3. <|im_end|> (ID=2):
    #      - 聊天消息结束符（End of Message）
    #      - 标记每条消息的结束
    #      - 也用作序列结束符（eos_token）
    #
    # 注意：这些 Token 的顺序非常重要！训练器会按照这个顺序分配 ID（0, 1, 2）
    #       后续的聊天模板和模型训练都依赖这些 ID，顺序错误会导致格式化失败
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # ========== 步骤 4：配置 BPE 训练器 ==========
    trainer = trainers.BpeTrainer(
        # vocab_size=6400: 目标词汇表大小
        #   - 包含 3 个特殊 Token，所以实际 BPE 学习的词表大小为 6397
        #   - 这个大小在编码效率和模型容量之间取得平衡
        #   - 较小的词汇表（如 3200）可能无法充分表达复杂文本
        #   - 较大的词汇表（如 12800）会增加模型参数和计算开销
        vocab_size=6400,
        # show_progress=True: 显示训练进度条
        show_progress=True,
        # special_tokens: 特殊 Token 列表，这些 Token 会被优先添加到词汇表中
        #   训练器会按照列表顺序为它们分配 ID（0, 1, 2, ...）
        special_tokens=special_tokens,
        # initial_alphabet: 初始字母表
        #   - ByteLevel.alphabet() 返回所有 256 个可能的字节值
        #   - 确保所有单字节都被包含在初始词汇表中
        #   - 这样可以避免字节级 OOV 问题，任何字节组合都能被编码
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # ========== 步骤 5：训练分词器 ==========
    # 从 JSONL 文件读取文本数据（生成器，节省内存）
    tests = read_texts_from_jsonl(data_path)
    # 使用迭代器训练分词器
    #   - train_from_iterator 会遍历所有文本
    #   - 统计字符对频率，执行 BPE 合并算法
    #   - 构建最终的词汇表（包含 6400 个 token）
    #   这个过程可能需要几分钟到几十分钟，取决于数据集大小
    tokenizer.train_from_iterator(tests, trainer=trainer)

    # ========== 步骤 6：设置解码器并验证特殊 Token ==========
    # 设置解码器：与 ByteLevel 预处理器对应
    #   - 解码器负责将 BPE token 序列转换回原始文本
    #   - ByteLevel 解码器会将字节序列重新组合为 UTF-8 字符串
    #   - 例如：[0xe4, 0xbd, 0xa0] → "你"
    tokenizer.decoder = decoders.ByteLevel()
    
    # 强制验证特殊 Token 的 ID（确保顺序正确）
    #   这些断言确保特殊 Token 的 ID 严格按照预期顺序：
    #   - <|endoftext|> 必须是 0（用作 pad_token 和 unk_token）
    #   - <|im_start|> 必须是 1（用作 bos_token）
    #   - <|im_end|> 必须是 2（用作 eos_token）
    #   如果验证失败，说明训练过程有问题，需要检查训练器配置
    #   后续的聊天模板和模型训练都依赖这些 ID，顺序错误会导致严重问题
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # ========== 步骤 7：保存分词器（关键：兼容 Hugging Face 格式） ==========
    # 保存的文件说明：
    #   1. tokenizer.json:
    #      - 分词器核心配置文件
    #      - 包含 BPE 模型参数、预处理/解码逻辑、完整词汇表
    #      - 这是 Hugging Face tokenizers 库的标准格式
    #
    #   2. vocab.json + merges.txt:
    #      - vocab.json: BPE 词汇表，包含所有 token 及其 ID 映射
    #      - merges.txt: BPE 合并规则，记录训练过程中学到的字符对合并顺序
    #      - 这两个文件由 tokenizer.model.save() 生成，是 BPE 算法的核心数据
    #
    #   3. tokenizer_config.json:
    #      - Transformers 库的兼容配置文件
    #      - 包含聊天模板、特殊 Token 映射、模型最大长度等
    #      - 这个文件需要手动创建（见下方代码）
    #      - 关键字段：
    #        * chat_template: 聊天消息格式化模板（Jinja2 格式）
    #        * pad_token/eos_token/bos_token: 特殊 Token 映射
    #        * model_max_length: 模型支持的最大序列长度（32768）
    #        * tokenizer_class: 指定使用 PreTrainedTokenizerFast 类加载
    tokenizer_dir = './model/'
    os.makedirs(tokenizer_dir, exist_ok=True)
    # 保存核心分词器配置
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    # 保存 BPE 模型（生成 vocab.json 和 merges.txt）
    tokenizer.model.save("./model/")

    # ========== 手动创建配置文件 tokenizer_config.json ==========
    # 这个配置文件是 Transformers 库加载分词器时必需的
    # 它定义了分词器的行为、特殊 Token 映射、聊天模板等关键信息
    config = {
        # add_bos_token: 是否自动在输入开头添加 BOS（Begin of Sequence）token
        #   设置为 False，因为我们使用 <|im_start|> 手动标记消息开始
        "add_bos_token": False,
        # add_eos_token: 是否自动在输入结尾添加 EOS（End of Sequence）token
        #   设置为 False，因为我们使用 <|im_end|> 手动标记消息结束
        "add_eos_token": False,
        # add_prefix_space: 是否在文本前添加空格
        #   设置为 False，因为中文等语言不需要前缀空格
        "add_prefix_space": False,
        # added_tokens_decoder: 特殊 Token 的解码映射
        #   键是 token 的 ID（字符串格式），值包含 token 的详细属性
        #   这个映射告诉 Transformers 如何解码这些特殊 token
        "added_tokens_decoder": {
            # ID=0: <|endoftext|> token 的配置
            "0": {
                "content": "<|endoftext|>",  # token 的实际文本内容
                "lstrip": False,  # 解码时不在左侧去除空格
                "normalized": False,  # 不进行 Unicode 规范化
                "rstrip": False,  # 解码时不在右侧去除空格
                "single_word": False,  # 不是单词语境（可以出现在词中间）
                "special": True  # 标记为特殊 token（不会被进一步分词）
            },
            # ID=1: <|im_start|> token 的配置
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            # ID=2: <|im_end|> token 的配置
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        # additional_special_tokens: 额外的特殊 token 列表（当前为空）
        "additional_special_tokens": [],
        # bos_token: 序列开始符，映射到 <|im_start|>
        #   在生成任务中，模型会使用这个 token 作为序列的开始
        "bos_token": "<|im_start|>",
        # clean_up_tokenization_spaces: 是否清理分词后的空格
        #   设置为 False，保持原始格式
        "clean_up_tokenization_spaces": False,
        # eos_token: 序列结束符，映射到 <|im_end|>
        #   在生成任务中，模型生成这个 token 时表示序列结束
        "eos_token": "<|im_end|>",
        # legacy: 是否使用旧版兼容模式
        #   设置为 True 以确保与旧版本 Transformers 兼容
        "legacy": True,
        # model_max_length: 模型支持的最大序列长度（token 数）
        #   设置为 32768，表示模型可以处理最长 32768 个 token 的序列
        #   这个值需要根据实际模型架构调整
        "model_max_length": 32768,
        # pad_token: 填充符，映射到 <|endoftext|>
        #   在批处理时，较短的序列会用这个 token 填充到相同长度
        "pad_token": "<|endoftext|>",
        # sp_model_kwargs: SentencePiece 模型参数（BPE 不使用，留空）
        "sp_model_kwargs": {},
        # spaces_between_special_tokens: 特殊 token 之间是否添加空格
        #   设置为 False，保持紧凑格式
        "spaces_between_special_tokens": False,
        # tokenizer_class: 指定 Transformers 加载时使用的分词器类
        #   PreTrainedTokenizerFast 是快速分词器类，基于 Rust 实现，性能更好
        "tokenizer_class": "PreTrainedTokenizerFast",
        # unk_token: 未知词 token，映射到 <|endoftext|>
        #   当遇到词汇表中不存在的词时使用（BPE 理论上不会有未知词，但保留此配置）
        "unk_token": "<|endoftext|>",
        # chat_template: 聊天消息格式化模板（Jinja2 语法）
        #   这是本分词器最核心的配置之一，定义了如何将多轮对话转换为模型输入
        #
        # 模板功能说明：
        #   1. 支持工具调用（tools）：如果提供了工具定义，会格式化工具调用提示
        #   2. 支持系统消息：处理 system role 的消息
        #   3. 支持多轮对话：格式化 user、assistant、tool 等不同角色的消息
        #   4. 消息格式：<|im_start|>role\ncontent<|im_end|>\n
        #   5. 生成提示：如果 add_generation_prompt=True，会在末尾添加 <|im_start|>assistant\n
        #
        # 使用示例：
        #   messages = [
        #       {"role": "system", "content": "你是一个助手"},
        #       {"role": "user", "content": "你好"},
        #       {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        #   ]
        #   prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        #   输出：
        #   <|im_start|>system
        #   你是一个助手<|im_end|>
        #   <|im_start|>user
        #   你好<|im_end|>
        #   <|im_start|>assistant
        #   你好！有什么可以帮助你的？<|im_end|>
        #
        # 注意：这个模板是 Jinja2 格式，支持条件判断、循环等复杂逻辑
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    # 保存配置文件到 tokenizer_config.json
    #   这个文件是 Transformers 库加载分词器时必需的配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        # ensure_ascii=False: 允许保存中文字符（不转义为 \uXXXX）
        # indent=4: 使用 4 空格缩进，提高可读性
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

def eval_tokenizer():
    """
    训练完成后，用 transformers 加载并验证分词器的核心功能。
    
    验证内容：
    1. 分词器加载：验证能否正确加载保存的分词器文件
    2. 聊天模板：验证聊天模板能否正确格式化多轮对话
    3. 编解码一致性：验证编码后再解码是否能还原原始文本
    4. 词汇表大小：验证词汇表大小是否符合预期（6400）
    
    这个函数主要用于调试和验证，确保分词器训练和保存过程正确无误。
    如果验证失败，说明训练或保存过程有问题，需要检查。
    """
    # ========== 步骤 1：加载分词器 ==========
    # 使用 Transformers 的 AutoTokenizer 加载保存的分词器
    #   这会自动读取 ./model/ 目录下的以下文件：
    #   - tokenizer.json: 核心分词器配置
    #   - tokenizer_config.json: Transformers 兼容配置
    #   - vocab.json 和 merges.txt: BPE 词汇表和合并规则
    from transformers import PreTrainedTokenizerFast
    # 注意：这里使用相对路径 "./model/"，在实际使用时建议使用绝对路径
    #   如果从不同目录运行脚本，相对路径可能找不到文件
    tokenizer = AutoTokenizer.from_pretrained("./model/")

    # ========== 步骤 2：验证聊天模板 ==========
    # 创建测试用的多轮对话消息
    #   包含 system、user、assistant 三种角色，模拟真实的对话场景
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # 使用聊天模板格式化消息
    #   apply_chat_template 会使用 tokenizer_config.json 中的 chat_template
    #   将消息列表转换为模型输入格式
    new_prompt = tokenizer.apply_chat_template(
        messages,
        # tokenize=False: 直接返回格式化后的字符串，而不是 token ID 序列
        #   这样可以直观地看到格式化结果，便于验证模板是否正确
        tokenize=False
    )
    # 打印格式化后的结果，应该看到类似：
    #   <|im_start|>system
    #   你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
    #   <|im_start|>user
    #   你来自哪里？<|im_end|>
    #   <|im_start|>assistant
    #   我来自地球<|im_end|>
    print(new_prompt)

    # ========== 步骤 3：验证词汇表大小和编解码一致性 ==========
    # 验证词汇表大小
    #   len(tokenizer) 返回词汇表的大小，应该是 6400（包含 3 个特殊 token）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)
    # 预期输出：tokenizer实际词表长度： 6400

    # 编码测试：将文本转换为 token ID 序列
    #   tokenizer() 会返回一个字典，包含 'input_ids'、'attention_mask' 等字段
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    # 输出编码后的 token 数量

    # 解码测试：将 token ID 序列转换回文本
    #   验证编码-解码的往返一致性
    input_ids = model_inputs['input_ids']
    # skip_special_tokens=False: 保留特殊 token（<|im_start|>, <|im_end|> 等）
    #   设置为 False 以便完整还原原始格式
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    # 验证解码后的文本是否与原始格式化文本完全一致
    #   如果一致，说明分词器的编码和解码逻辑正确
    print('decoder和原始文本是否一致：', response == new_prompt)
    # 预期输出：decoder和原始文本是否一致： True

def main():
    """
    主函数：执行分词器训练流程。
    
    流程：
    1. 调用 train_tokenizer() 训练并保存分词器
    2. （可选）调用 eval_tokenizer() 验证分词器功能
    
    注意：
        - eval_tokenizer() 默认被注释，因为验证需要分词器已经训练完成
        - 如果需要验证，取消注释 eval_tokenizer() 即可
        - 验证功能主要用于调试，确保训练过程正确
    """
    # 训练分词器：从数据读取到保存完整流程
    train_tokenizer()
    # 验证分词器：取消注释以验证训练结果
    #   注意：验证时需要确保 ./model/ 目录存在且包含完整的分词器文件
    #   如果从不同目录运行脚本，可能需要修改 eval_tokenizer() 中的路径
    eval_tokenizer()

if __name__ == "__main__":
    # 当脚本直接运行时（而非被导入），执行主函数
    main()
```





















