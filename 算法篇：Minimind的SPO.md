# 算法篇：Minimind的SPO

## 引言

本文经Gemini润色得来，如有AI味纯属故意~

在前面的学习中，我们已经深入剖析了大模型在强化学习（RL）对齐阶段的两种主流方案。传统 PPO 虽然是行业基石，但其包含 Actor、Critic、Reference 和 Reward 四个模型的架构，对显存提出了极其苛刻的要求。为了解决显存瓶颈，GRPO 的出现，通过“组内相对优势”成功移除了 Critic 模型，但它依赖群体采样（Group Sampling），通常需要对同一个 Prompt 生成 4 到 8 个不同的回答，这在无形中成倍地增加了推理生成的算力消耗。

**那么，有没有一种方法能够既减少各个模型的数量，又减少采样消耗呢？有的兄弟，有的！**

SPO 的核心特点可以用一句话概括：**它像 GRPO 一样不使用 Critic 模型（省显存），同时又像 PPO 一样每次只对 Prompt 采样 1 个回答（省算力）。**

首先用通俗易懂的话建立对SPO的大致印象：

*SPO 在后台用数学公式维护了一个**“历史平均分”**（自适应价值追踪器）。每次模型做完题，SPO 就直接拿这次的真实得分去和“历史平均分”比，比平均分高就鼓励，比平均分低就惩罚。同时，为了防止模型为了刷分而突然“走火入魔”（策略突变），它还会实时监控模型回答风格的变化幅度（KL 散度），一旦发现变化太剧烈，就会在数学层面上“踩一脚刹车”，让训练平稳进行。*

下面，我们将详细拆解 SPO 的完整代码逻辑。

---

## 1 算法核心： `AutoAdaptiveValueTracker`

在强化学习中，计算优势（Advantage）需要一个“基准线（Baseline）”。PPO 用神经网络（Critic）去猜这个基准线，GRPO 用组内所有人的平均分作为基准线。

SPO 则是用纯数学的**指数移动平均（EMA）追踪器**来记录历史得分，以此作为基准线。这就是 `AutoAdaptiveValueTracker` 类的作用。

### 1.1 初始化与 Beta 分布追踪

```python
class AutoAdaptiveValueTracker:
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode         # rho 计算模式，默认使用基于 KL 的动态衰减
        self.rho_const = rho_const       # 固定模式下的衰减常数
        self.D_half = D_half             # KL 散度的半衰期。当 KL = 0.06 时，动量 rho 减半
        self.clip_lower = clip_lower     # 动量 rho 的下限 (0.5)
        self.clip_upper = clip_upper     # 动量 rho 的上限 (0.96)
        
        N_init = 1.0 / (1.0 - self.clip_lower) # 初始化伪计数
        self.alpha = 0.5 * N_init        # Beta 分布的 alpha 参数
        self.beta = 0.5 * N_init         # Beta 分布的 beta 参数
        self.old_mean_logprob = None     # 用于记录上一轮的平均对数概率
```

以下是Gemini生成的关于每个参数的详细解释：

#### 1.1.1 模式与常数控制

`self.rho_mode = rho_mode`

- **如何使用**：在 `compute_rho()` 函数中充当“开关”。
- **具体逻辑**：如果它的值是 `'constant'`，代码就会跳过所有复杂的 KL 散度计算，直接返回固定的动量值。默认它是 `'kl'`，表示开启**动态动量衰减**，让算法根据模型状态自动踩刹车。

`self.rho_const = rho_const` (默认 0.9)

- **如何使用**：作为算法的“保底”或“静态”动量（Momentum）。
- **具体逻辑**：
  1. 当 `rho_mode == 'constant'` 时，永远返回这个值。
  2. 在训练的**第一步**（此时还没有上一轮的历史概率 `old_mean_logprob`），无法计算 KL 散度，就会默认返回这个 0.9。
  3. 它的作用是决定历史基线的“粘性”。在 EMA（指数移动平均）更新时，新基线会保留 90% 的历史记忆，仅吸收 10% 的当前新数据。

------

#### 1.1.2 动态衰减的核心物理量

`self.D_half = D_half` (默认 0.06)

- **如何使用**：在 `compute_rho()` 函数中作为指数函数的**半衰期阈值**。
- **具体逻辑**：它直接参与计算公式 $\rho = 2^{-\text{kl} / D_{half}}$。
  - 这里的 `kl` 是新旧策略概率均值的变化量。
  - 当模型的策略偏移量（KL）恰好等于 0.06 时，指数部分变为 $-1$，此时 $\rho = 2^{-1} = 0.5$。
  - 这意味着：**当模型的生成概率分布相比上一步发生了 0.06 的偏移时，历史经验的保留权重就会“减半”（降到极低值）**，迫使基线快速吸收当前的新状态，防止基线与模型实际能力脱节。

------

#### 1.1.3 安全裁剪边界 (Clipping)

`self.clip_lower = clip_lower` (默认 0.5)

- **如何使用**：用于限制 $\rho$ 的下限，并用于初始化伪计数。
- **具体逻辑**：
  1. 在 `compute_rho` 的最后：`max(..., self.clip_lower)`。无论模型震荡多么剧烈，KL 散度有多大，$\rho$ 最小只能跌到 0.5。如果跌到 0，意味着彻底抛弃历史均值，这会导致基线随单个 Batch 剧烈跳变，失去“均值”的意义。
  2. 用于计算 `N_init`。

`self.clip_upper = clip_upper` (默认 0.96)

- **如何使用**：用于限制 $\rho$ 的上限。
- **具体逻辑**：在 `compute_rho` 的最后：`min(..., self.clip_upper)`。即使模型毫无变化（KL 为 0，理论上 $\rho$ 会等于 1.0），也强制截断在 0.96。如果 $\rho = 1.0$，基线更新公式中的新奖励权重 $(1 - \rho)$ 就会变成 0，导致**基线彻底冻结**，永远不吸收新知识。设定 0.96 保证了无论如何都会有 4% 的新经验被融入。

------

#### 1.1.4 Beta 分布的核心追踪变量

`N_init = 1.0 / (1.0 - self.clip_lower)`

- **如何使用**：这是一个局部变量，用于推导 $\alpha$ 和 $\beta$ 的初始值。
- **具体逻辑**：这在统计学中被称为**伪计数（Pseudo-count）**。代入 0.5 后，$N_{init} = 2.0$。它相当于假设我们在训练开始前，就已经“虚构”地观察到了 2 个样本，防止初始的少数几个 Batch 把基线带偏。

`self.alpha = 0.5 * N_init`

- **如何使用**：在 `get_baselines()` 中计算基准分，在 `update()` 中累加正向奖励。
- **具体逻辑**：
  - 初始值为 $0.5 \times 2.0 = 1.0$。它代表历史累积的**正向得分**。
  - 更新公式：$\alpha_{new} = \rho \cdot \alpha_{old} + \text{reward}_{norm}$。每训练一步，就把当前 Batch 的归一化得分加进去。

`self.beta = 0.5 * N_init`

- **如何使用**：与 `alpha` 配合，代表历史累积的**负向得分**。
- **具体逻辑**：
  - 初始值同样为 1.0。
  - 输出基准线：在 `get_baselines` 中，每次算基准分就是直接求期望值：$\text{Baseline} = \frac{\alpha}{\alpha + \beta}$。
  - 更新公式：$\beta_{new} = \rho \cdot \beta_{old} + (1 - \text{reward}_{norm})$。

------

#### 1.1.5 记忆变量

`self.old_mean_logprob = None`

- **如何使用**：用于跨 Batch 计算 KL 散度。
- **具体逻辑**：
  - 在 `update()` 时，它会接收并保存当前 Batch 模型输出的**平均对数概率**。
  - 在下一个 Batch 进来时，`compute_rho()` 会把它拿出来，与新的对数概率做减法求绝对值：$\text{KL} = |\text{old\_mean\_logprob} - \text{cur\_mean\_logprob}|$。
  - 可以说，它是算法感知模型“是否正在发生突变”的唯一记忆锚点。

---

### 1.2 计算动量衰减因子 $\rho$（Rho）

如果在训练中模型突然策略发生剧烈偏移，我们要如何防止基线被带偏？SPO 放弃了 PPO 中生硬的 `torch.clamp`，转而使用动态计算的动量 $\rho$（Rho）。这个 $\rho$ 决定了我们在评估模型表现时，是应该“多看看历史平均分”，还是应该“赶紧忘掉历史，多看看眼前的新分数”。

```python
def compute_rho(self, cur_mean_logprob):
    if self.rho_mode == 'constant':
              return self.rho_const
    if self.old_mean_logprob is None:
        return self.rho_const

    # 计算新旧策略概率均值的绝对差（简化的单标量 KL 散度）
    kl = abs(self.old_mean_logprob - cur_mean_logprob)
    # 根据 KL 散度计算动量 rho：KL 越大，rho 越小
    rho = 2 ** (-kl / self.D_half)

    return max(min(rho, self.clip_upper), self.clip_lower)
```

我们一步一步讲解，把它的物理意义和数学逻辑拆解开来：

#### 1.2.1 冷启动处理

```python
if self.old_mean_logprob is None:
        return self.rho_const
```

这是应对训练刚刚开始（或追踪器刚被重置）时的逻辑。要计算模型策略“变化了多少”，我们必须有“上一次”的记录（`old_mean_logprob`）。但第一步训练时，我们还没有历史记录。此时无法计算变化量，所以代码直接返回一个在初始化时设定好的默认常数 `self.rho_const`。这意味着在最初阶段，我们默认给历史经验分配 90% 的权重。

#### 1.2.2 计算策略偏移量（变化探测器）

```python
    # 计算新旧策略概率均值的绝对差（简化的单标量 KL 散度）
    kl = abs(self.old_mean_logprob - cur_mean_logprob)
```

测量模型在这一步更新中，策略变了多少。

- `cur_mean_logprob` 是模型当前生成这段话的平均置信度（对数概率）。
- `old_mean_logprob` 是模型在上一个 Batch 时的置信度。
- 两者相减并取绝对值，是对 **KL 散度（KL Divergence）** 的一种极其轻量、简化的标量近似。如果 `kl` 很小，说明模型只是在微调；如果 `kl` 很大，说明模型的输出策略正在发生剧烈的跳变。

------

#### 1.2.3 半衰期衰减计算（核心刹车机制）

```python
    # 根据 KL 散度计算动量 rho：KL 越大，rho 越小
    rho = 2 ** (-kl / self.D_half)
```

根据刚才算出的变化量，动态决定历史记录的保留比例 $\rho$。

这是一个**指数半衰期公式**。`self.D_half` 是设定的半衰期阈值（例如 0.06）。**假设模型变化很大**：如果算出的 `kl` 刚好等于 0.06，那么指数部分就是 $-1$，此时 $\rho = 2^{-1} = 0.5$。这意味着历史记忆的权重瞬间被“腰斩”了。**为什么要这样做？** 如果模型策略发生了剧变，那么**过去的平均分就已经不准了**（不能用来客观评价现在的模型）。此时调低 $\rho$，能让追踪器更快地“遗忘”旧的平均分，迅速吸收当前的新分数。新平均分迅速跟上后，算出的“优势值（Advantage）”就会变小，从而在数学上**拉住模型的梯度，防止它继续朝着偏离的方向移动**。这就是 SPO 不需要 `clip` 截断也能稳定训练的原因。

------

#### 1.2.4 上下限截断

```python
    return max(min(rho, self.clip_upper), self.clip_lower)
```

给 $\rho$ 加上强制的上下限安全网，防止极端情况搞崩训练。**`min(rho, self.clip_upper)`（防冻结）**：假设模型完全没变（`kl` = 0），理论上 $\rho$ 会等于 1.0。如果 $\rho$ 是 1.0，历史权重就是 100%，系统将永远不再吸收任何新的奖励。所以代码用 `clip_upper`（如 0.96）卡住上限，保证每一次多多少少都会更新一点新知识。**`max(..., self.clip_lower)`（防失忆）**：假设模型发生了极其离谱的突变，导致 `kl` 极大，$\rho$ 会无限趋近于 0。如果 $\rho$ 变成 0，历史积累的基准线就会被瞬间清空，导致接下来的评分完全随机波动。所以用 `clip_lower`（如 0.5）托底，保证无论怎么变，至少要保留一半的历史记忆，维持评分系统最基本的稳定性。

---

### 1.3 基线的获取与更新

它在每一个训练 Batch（批次）结束时被调用，负责将模型刚才取得的成绩记录到历史档案中，从而更新我们的“全局平均分（基线）”。$\rho$ 决定了“历史经验”的权重。

```python
def update(self, rewards, cur_logprobs=None, response_masks=None):
    # 1. 计算当前的平均对数概率，并据此计算 rho
    if cur_logprobs is not None and response_masks is not None:
        # 仅对模型生成的 Response 部分求平均概率
        mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
        rho = self.compute_rho(mean_logprob)
        self.old_mean_logprob = mean_logprob # 缓存当前概率留作下次用
    else:
        rho = self.rho_const

    # 2. 奖励归一化
    scale = 3.0
    # 将原本 [-3.0, 3.0] 的奖励，线性映射到 [0, 1] 区间
    normalized_rewards = (rewards + scale) / (2 * scale)
    avg_normalized_reward = normalized_rewards.mean().item()

    # 3. 指数移动平均 (EMA) 更新 alpha 和 beta
    # 过去积累的经验乘以 rho（通常是 0.9 左右），加上当前的新经验
    self.alpha = rho * self.alpha + avg_normalized_reward
    self.beta = rho * self.beta + (1 - avg_normalized_reward)
    return rho
```

#### 1.3.1 感知策略变化，决定动量 ($\rho$)

```python
    # 1. 计算当前的平均对数概率，并据此计算 rho
    if cur_logprobs is not None and response_masks is not None:
        # 仅对模型生成的 Response 部分求平均概率
        mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
        rho = self.compute_rho(mean_logprob)
        self.old_mean_logprob = mean_logprob # 缓存当前概率留作下次用
    else:
        rho = self.rho_const
```

- **为什么需要 `response_masks`？** 大模型生成的句子有长有短，在组成一个 Batch 时，短句子后面会填充无意义的 `<pad>` 字符；而且输入中还包含用户提问的 Prompt。我们评估模型的置信度时，**不能**把 Prompt 和 `<pad>` 的概率算进去。所以 `cur_logprobs * response_masks` 就是用掩码把无关的 token 屏蔽掉（变成0），然后再求和、除以有效 token 数量，得出纯正的“回答部分的平均置信度”。

- **动态刹车 `compute_rho`**：

  拿到当前的置信度后，立刻去跟上一步缓存的置信度做对比（这就是上文提到的 KL 散度探测）。算出本次更新的遗忘/保留系数 $\rho$。

- **存档**：

  算出 $\rho$ 后，当前这一步的 `mean_logprob` 就变成了历史，存入 `self.old_mean_logprob`，留给下一个 Batch 用来算 KL。

------

#### 1.3.2 奖励归一化

```python
    # 2. 奖励归一化
    scale = 3.0
    # 将原本 [-3.0, 3.0] 的奖励，线性映射到 [0, 1] 区间
    normalized_rewards = (rewards + scale) / (2 * scale)
    avg_normalized_reward = normalized_rewards.mean().item()
```

- **为什么要归一化？**

  外部的奖励模型（Reward Model）或者规则打分给出的原始分数，通常是有正有负的。但是，SPO 底层使用的是 **Beta 分布** 的逻辑，Beta 分布是用来描述“概率”的，它**只接受 $0$ 到 $1$ 之间的数值**。

- **例子**：

  - 最差得分：$-3.0 \rightarrow (-3.0 + 3.0) / 6.0 = 0.0$
  - 中等得分：$0.0 \rightarrow (0.0 + 3.0) / 6.0 = 0.5$
  - 满分得分：$3.0 \rightarrow (3.0 + 3.0) / 6.0 = 1.0$

- **求批次均值**：

  将这个 Batch 里的所有分数取平均，得到 `avg_normalized_reward`。它代表了模型在**这个batch**上的总体表现有多好（比如 0.8，相当于得分为良）。

------

#### 1.3.3 更新历史总账，EMA 指数移动平均

```python
    # 3. 指数移动平均 (EMA) 更新 alpha 和 beta
    # 过去积累的经验乘以 rho（通常是 0.9 左右），加上当前的新经验
    self.alpha = rho * self.alpha + avg_normalized_reward
    self.beta = rho * self.beta + (1 - avg_normalized_reward)
    return rho
```

前面提到过，$\alpha$ 代表历史累积的“正向经验”（赢的次数），$\beta$ 代表历史累积的“负向经验”（输的次数）。

- **更新 $\alpha$（累积胜利）**：

  公式 `rho * old_alpha + avg_normalized_reward`。

  假设 $\rho$ 是 0.9，本次表现是 0.8。意思是：过去的胜利经验打个 9 折（保留 90%），然后加上本次的 0.8 分。

- **更新 $\beta$（累积失败）**：

  公式 `rho * old_beta + (1 - avg_normalized_reward)`。

  本次表现是 0.8，说明丢了 0.2 分（$1 - 0.8$）。所以过去的失败经验也打个 9 折，加上本次丢掉的 0.2 分。

为什么要这样设计？

1. **平滑性**：如果不乘以 $\rho$，$\alpha$ 和 $\beta$ 会无限变大，最终新加进去的分数就像一滴水落入大海，再也无法影响基线（模型停止学习）。
2. **时效性**：乘以小于 1 的 $\rho$（比如 0.9），意味着越久远的历史分数，会被乘越多次的 0.9，权重呈指数级衰减。这保证了**基线永远反映的是模型“最近一段时间”的真实水平**。
3. 如果模型在第一步算出的 $\rho$ 变小了（比如因为策略震荡降到了 0.5），这里的历史分数就会被严重打折，Tracker 会迅速接纳最新的 `avg_normalized_reward`，从而快速重置基线，阻止模型向错误的方向坠落。

最后，函数将计算好的 $\rho$ 返回，供主训练脚本（如果需要的话）记录到 wandb 日志中，以便开发者观察训练动态。

---

## 2 主训练循环 `spo_train_epoch`

这是整个算法执行的流水线。我们将按阶段进行拆解。

### 2.1 数据预处理与生成 (Rollout)

```python
def spo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, value_tracker, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # 提取 Prompt 文本列表，长度为 Batch_size (B)
        
        # 将文本转换为 token，注意 padding_side="left"，这是自回归生成的标准操作
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        
        # 截断超长的 Prompt，防止显存溢出
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad(): # 纯推理阶段，不计算梯度
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # 模型自回归生成。注意 num_return_sequences=1，与 GRPO 的多样本生成不同
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)  # 输出形状: [B, P+R]

        # 仅切片出模型自己生成的回复部分 (Response)，丢弃前面的 Prompt
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # 形状: [B, R]
```

#### 2.1.1 提取文本与“左侧填充” (Left Padding)

```Python
        prompts = batch['prompt']  # 提取 Prompt 文本列表，长度为 Batch_size (B)
        
        # 将文本转换为 token，注意 padding_side="left"，这是自回归生成的标准操作
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
```

拿到 DataLoader 给的纯文本问题，用 Tokenizer 将其变成模型认识的数字 ID，并打包成张量送到显卡上。

- **注意 `padding_side="left"`**：在组成一个 Batch 时，因为每句话长度不同，必须用无意义的 `<pad>` 字符把短句补齐到和最长句一样长。
  - **如果右侧填充 (Right Padding)**：`[问题 A, <pad>, <pad>]`。模型自回归生成时，会在序列末尾接着写，也就是在 `<pad>` 后面续写，这会导致位置编码完全错乱，生成的文字变成乱码。
  - **所以必须左侧填充 (Left Padding)**：`[<pad>, <pad>, 问题 A]`。这样所有句子的**最后一个有效词都在右侧对齐**，模型可以直接顺着右侧往后生成新词。

#### 2.1.2 左侧截断防溢出

```Python
        # 截断超长的 Prompt，防止显存溢出
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

强制把问题长度卡在一个上限（比如 1024 个 token）内，防止某个恶意超长 Prompt 直接把显存撑爆（OOM）。

- **为什么是 `[:, -args.max_seq_len:]`？** 注意前面的负号 `-`。因为我们刚才使用了“左侧填充”，真实有效的问题内容都在**序列的末尾（右侧）**。如果从左边正向截断 `[:max_len]`，可能会把真正的问题截没，只留下一堆 `<pad>`。从右向左截取（提取倒数 `max_seq_len` 个 token），就能保证保留的都是距离生成位置最近、最重要的上下文。

#### 2.1.3 无梯度自回归生成 (Rollout)

```Python
        with torch.no_grad(): # 纯推理阶段，不计算梯度
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # 模型自回归生成。注意 num_return_sequences=1，与 GRPO 的多样本生成不同
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)  # 输出形状: [B, P+R]
```

- **`torch.no_grad()`**：在这里模型只是在前向推理，不是在反向传播更新参数，所以不需要保留复杂的梯度计算图。
- **DDP 解包**：如果在多卡分布式（DDP）环境下，模型会被包上一层壳（`DistributedDataParallel`），它会屏蔽掉底层的 `generate` 方法。所以需要用 `.module` 把原始模型剥离出来才能调用生成函数。
- **`num_return_sequences=1`**：这是 SPO 区别于 GRPO 的核心参数。对于同一个问题，模型只做 1 遍，生成 1 个答案。
- **输出维度 `[B, P+R]`**：`generate` 函数默认会把**你输入的问题 (P)** 和 **模型自己生成的回答 (R)** 拼在一起作为完整的上下文返回。

#### 2.1.4 提取纯净答案

```Python
        # 仅切片出模型自己生成的回复部分 (Response)，丢弃前面的 Prompt
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # 形状: [B, R]
```

把刚才 `[B, P+R]` 形状的完整输出切开，把前面的问题（Prompt）扔掉，只保留模型自己生成的纯答案（Response）。

- `prompt_inputs["input_ids"].size(1)` 就是这个 Batch 里问题的总长度（包含了 pad）。利用切片 `[:, 长度:]`，准确地截取出从问题结束的下一个词开始，直到序列末尾的所有生成词。形状变成 `[B, R]`（Batch Size, Response Length）。
- **为什么要切分？** 因为在强化学习的计算中，我们只对模型**自己生成的行为（动作）**计算对数概率并给予奖惩，用户输入的问题部分是固定状态，不需要也不应该参与策略梯度的更新。

---

### 2.2 提取 Token 级对数概率

这段代码分为两部分：定义提取概率的内部函数 `get_per_token_logps`，以及分别调用它来获取“当前模型（Actor）”和“参考模型（Reference）”的概率。我们逐一拆解。

```python
        def get_per_token_logps(mdl, input_ids, n_keep):
            # 推理模式下需要克隆张量，防止 inplace 修改报错
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            
            # 1. 前向传播获取所有 token 的原始 Logits
            # 切片 [:, :-1, :]：极其关键的错位操作！丢弃最后一个预测结果，使得 Logits 对应的预测位置和 input_ids 对齐
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            
            per_token_logps = []
            # 遍历 Batch 中的每一行
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]): # 只取生成的 Response 部分
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                
                # 2. log_softmax 将 Logits 转化为对数概率
                # 3. gather 操作：拿着模型实际生成的 Token ID (ids_row)，去概率分布中把对应的概率提取出来
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            
            # 将列表堆叠回 Tensor
            return torch.stack(per_token_logps) # 形状: [B, R]

        # 计算 Actor 模型（有梯度）的生成概率
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B, R]
            res = model(outputs) if lm_config.use_moe else None # 如果是 MoE，额外算一次拿辅助 Loss
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        # 计算 Ref 模型（无梯度）的生成概率
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B, R]
```

#### 2.2.1 `get_per_token_logps`

这个函数的任务是：输入一段完整的句子（Prompt + Response），输出仅仅属于 Response 部分每个字的生成概率。

```Python
def get_per_token_logps(mdl, input_ids, n_keep):
    # 推理模式下需要克隆张量，防止 inplace 修改报错
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
```

这是 PyTorch 的工程细节。大模型在做多步自回归生成后，底层的 Tensor 可能会共享内存或处于某些特殊的推理视图下。为了防止后续计算梯度时发生 `Inplace Operation` 报错，这里做了一次安全的克隆拷贝。

**错位切片**

```Python
    # 1. 前向传播获取所有 token 的原始 Logits
    # 切片 [:, :-1, :]：极其关键的错位操作！丢弃最后一个预测结果，使得 Logits 对应的预测位置和 input_ids 对齐
    logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
```

前面的文章已经把这里的逻辑降了很多遍了，不过还是再讲讲以加深印象。

假设我们的输入 `input_ids` 是一句话：“[A, B, C, D]”。

模型前向传播输出的 `logits`（原始预测分数）实际上是：

- 输入 A 时，输出对 **B** 的预测。
- 输入 B 时，输出对 **C** 的预测。
- 输入 C 时，输出对 **D** 的预测。
- 输入 D 时，输出对 **下一个未知词** 的预测。

如果我们想知道模型生成 [B, C, D] 的概率，我们需要拿到前三个 Logits。所以，**`[:, :-1, :]` 这个切片操作，直接把最后一个没有用的“对未知词的预测”给丢弃了**。丢弃后，Logits 的数量恰好和预测出的 Token 数量一一对应。

*（注：`logits_to_keep=n_keep + 1` 是底层的显存优化。因为我们只关心生成的 `n_keep` 个词，不需要计算 Prompt 部分的梯度，所以让模型只返回最后 `n_keep + 1` 个 Logits，然后切掉最后一个，刚好剩下 `n_keep` 个。）*

 **`gather` 操作**

```Python
    per_token_logps = []
    # 遍历 Batch 中的每一行
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]): # 只取生成的 Response 部分
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        
        # 2. log_softmax 将 Logits 转化为对数概率
        # 3. gather 操作：拿着模型实际生成的 Token ID (ids_row)，去概率分布中把对应的概率提取出来
        per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
```

- `ids_row`：这是模型刚才实际生成的句子。比如模型生成了词汇表里 ID 为 `[512, 1024, 256]` 的三个词。
- `logits_row.log_softmax(dim=-1)`：模型的原始输出包含了对词汇表里几万个词的打分（比如维度是 `[R, 词表大小]`）。这步将其转化为严谨的对数概率分布 $\log \pi(x)$。
- **`torch.gather`**：这是精髓。模型给几万个词都打了分，但我们**只关心模型实际说出口的那个词的概率**。`gather` 的作用就像查字典：拿着实际生成的 ID `512`，去几万维的概率向量里，精准地把第 `512` 个位置的概率值抠出来。
- 最后，把这些抠出来的标量堆叠成一个张量返回，形状为 `[B, R]`（Batch大小 × 回答长度）。

------

#### 2.2.2 提取对数概率（Actor 与 Reference）

刚才说的是提取概率的用的函数，我们现在分别在actor模型和ref模型上调用它

```Python
        # 计算 Actor 模型（有梯度）的生成概率
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B, R]
            res = model(outputs) if lm_config.use_moe else None # 如果是 MoE，额外算一次拿辅助 Loss
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
```

- **Actor 模型（策略模型）**：这就是我们正在训练的 `model`。注意，这里**没有** `torch.no_grad()`。因此，算出来的 `per_token_logps` 是**带有梯度计算图的**（`requires_grad=True`）。一会我们计算完优势（Advantage）后，直接对它进行 `.backward()`，就能更新大模型的权重了。
- **MoE 辅助损失**：如果使用了混合专家（MoE）架构，还需要额外算一次前向传播来获取负载均衡损失（Aux Loss），防止所有数据都挤在一个专家头上。

```Python
        # 计算 Ref 模型（无梯度）的生成概率
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B, R]
```

为了防止模型训崩，我们需要知道“在训练之前，原来的模型原本是以多大概率说出这句话的？” 这就是 `ref_per_token_logps`。在后续算 Loss 时，我们会计算当前概率和参考概率的差值（KL散度）。如果当前模型偏离老模型太远，即使得分高，也会被强行扣除惩罚分，从而保证模型在提升能力的同时，不丧失基本的说话常识。因为不更新它，所以包在 `torch.no_grad()` 里以节省显存。

---

### 2.3 计算优势 (Advantage)

```python
        # 解码回文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  
        # 调用外部裁判模型打分
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B]

        # 从 Tracker 获取全局基线 (范围 [0, 1])
        baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]

        scale = 3.0
        # 将基线反归一化，恢复到与 Reward 相同的 [-3.0, 3.0] 量纲
        unnormalized_baselines = baselines * (2 * scale) - scale  # [B]
        
        # 计算优势：实际得分 - 预期得分
        advantages = rewards - unnormalized_baselines  # [B]

        # 简单粗暴的裁剪，防止模型因为极端的 Reward 而梯度爆炸
        advantages = advantages.clamp(-5.0, 5.0)
```

它的最终目的是算出一个**优势值（Advantage）**。

在强化学习中，光知道“得分高低”是不够的，模型需要知道的是“这次的发挥比平时的平均水平好多少或差多少”。这段代码就是完成这一逻辑的完整流水线。我们逐行进行拆解：

#### 2.3.1 将模型生成的 Token 还原为人类文字

```Python
# 解码回文本
completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  
```

- **为什么要做这一步？** 在上一阶段生成的 `completion_ids` 是一堆密密麻麻的数字（Token ID）。虽然大模型在底层算概率时用的是数字，但是**外部的裁判（奖励模型或基于正则的格式检查器）通常只认识人类文本**。
- `batch_decode` 会把这些数字翻译回它原本生成的文本串（例如：“我认为天空是蓝色的，因为...”）。`skip_special_tokens=True` 会把填充符 `<pad>` 或序列结束符 `<eos>` 去掉，保证送去打分的是纯净的文本。

#### 2.3.2 计算 Reward

```Python
# 调用外部裁判模型打分
rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B]
```

- 拿着问题（`prompts`）和模型刚写的答案（`completions`），交给 `calculate_rewards` 函数。
- 在 MiniMind 的设定中，这个函数会做两件事：
  1. **格式检查**：如果你开启了推理模式（`args.reasoning == 1`），它会用正则表达式检查回答里有没有包含 `<think>` 和 `<answer>` 标签。如果没有，直接扣分。
  2. **语义打分**：它会把问答对送入外部的一个 Reward Model（比如 InternLM2-Reward 模型）中，让这个专业的“打分专家”给出一个具体的分数。
- 返回的 `rewards` 是一个形状为 `[B]` 的一维张量（Batch 中每个问题对应一个最终得分）。

#### 2.3.3 获取历史平均Baseline

```Python
# 从 Tracker 获取全局基线 (范围 [0, 1])
baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]
```

- 分数拿到手了，但怎么判断这个分数是好是坏呢？比如考了 80 分，如果全班平均分是 90，那就是考差了；如果平均分是 60，那就是考好了。
- 这里调用了我们在前面解析过的 `value_tracker`。它通过内部的历史 Beta 分布记录，给出了模型最近一段时间的“平均预估得分”。此时拿到的基线分数范围被压缩在 **0 到 1** 之间。

#### 2.3.4 统一量纲（反归一化）

```Python
scale = 3.0
# 将基线反归一化，恢复到与 Reward 相同的 [-3.0, 3.0] 量纲
unnormalized_baselines = baselines * (2 * scale) - scale  # [B]
```

- 外部 Reward Model 给出的 `rewards` 在 **-3.0 到 3.0** 之间。而刚才 Tracker 给出的基线在 **0 到 1** 之间。单位不同不能直接相减。
- 这里做了一个简单的线性映射倒推运算：将 `[0, 1]` 乘以 6（即 `2 * scale`），变成 `[0, 6]`，再减去 3（即 `scale`），成功将其恢复到 **[-3.0, 3.0]** 的区间，使得基线与实际奖励处于同一个测量标尺下。

#### 2.3.5 核心计算：优势估计（Advantage）

```Python
# 计算优势：实际得分 - 预期得分
advantages = rewards - unnormalized_baselines  # [B]
```

- **如果优势为正（Advantage > 0）**：说明模型这次生成的回答比它的历史平均水平要好。在随后的反向传播中，Loss 公式会**推高**生成这句回答里各个 Token 的概率。
- **如果优势为负（Advantage < 0）**：说明这次发挥失常，写了一堆废话或格式错误。Loss 公式会**压低**这些 Token 的生成概率，让模型以后少犯这种错误。

#### 2.3.6 安全截断

```Python
advantages = advantages.clamp(-5.0, 5.0)
```

- 如果某一次 Reward Model 抽风，给了一个极其离谱的超高分或超低分，算出来的 `advantages` 就会极大，直接导致梯度爆炸，把模型的参数更新成 `NaN`（坏死）。
- 它强行规定：无论你这次考得有多好或多差，优势值最大不能超过 5.0，最小不能低于 -5.0。这保证了模型梯度的平稳更新。

---

### 2.4 构造掩码 (Masking)

```python
        # 寻找结束符 EOS 的位置
        is_eos = completion_ids == tokenizer.eos_token_id  # [B, R]的布尔矩阵
        
        # 初始化一个默认指向序列末尾的索引数组
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)  # [B]
        
        # 如果序列中有 EOS，就把索引更新为第一个出现 EOS 的位置
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # 构建有效 Mask：在 EOS 之前的位置为 1，之后填充的 Pad 为 0
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B, R]
```

在大模型的批量（Batch）生成中，模型同时回答多道题。有的题回答得很简短（比如 10 个词），有的题回答得很长（比如 50 个词）。为了能把它们塞进同一个矩阵里用 GPU 并行计算，短句子的后面会被强制填充大量无意义的 `<pad>` 字符，直到和最长的句子一样长。 如果在算 Loss 和奖励时，把这些 `<pad>` 字符也算进去，模型的梯度就会被严重污染。**因此，我们需要通过寻找结束符（EOS, End of Sequence），为每一句话画一条“截止线”，线左边的有效词算入 Loss，线右边的废话全部归零。**

#### 2.4.1 寻找 EOS

```Python
# 寻找结束符 EOS 的位置
is_eos = completion_ids == tokenizer.eos_token_id  # [B, R]的布尔矩阵
```

- `completion_ids` 是模型生成的回复矩阵，形状为 `[B, R]`（Batch大小 $\times$ 回答长度）。
- 这行代码做了一个逐元素的判断，生成了一个同样形状为 `[B, R]` 的布尔矩阵（True/False）。
- **作用**：只要这个词是 EOS，对应位置就标记为 `True`，否则为 `False`。

------

#### 2.4.2 假设最坏情况（没有 EOS）

```Python
# 初始化一个默认指向序列末尾的索引数组
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)  # [B]
```

- `is_eos.size(0)` 是 Batch Size (`B`)，`is_eos.size(1)` 是最大长度 (`R`)。
- `torch.full` 制造了一个长度为 `B` 的一维向量，里面填满了最大长度 `R`。
- **作用**：这是在设定**保底的截止位置**。因为有时候模型特别啰嗦，直到达到了我们设置的最大生成长度 `max_gen_len`，它都没生成 EOS。这时候我们默认整句话从头到尾都是有效的，所以把截止线设在序列的最末尾 `R`。

------

#### 2.4.3 定位第一个 EOS 的位置

```Python
# 如果序列中有 EOS，就把索引更新为第一个出现 EOS 的位置
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
```

这行代码看起来最复杂，其实逻辑非常严密：

1. **`is_eos.any(dim=1)`**：它会检查这一个 Batch 里的每一句话，只要这句话里出现了至少一个 `True`（也就是有 EOS），就返回 `True`。它用来筛选出“正常结束的句子”。
2. **`is_eos.int().argmax(dim=1)`**：把布尔值变成 0 和 1 后，`argmax` 会寻找这一行里**最大值（即 1）第一次出现的位置索引**。因为模型可能会在后面生成一堆乱码或多个 EOS，我们只认**第一个** EOS！
3. **整体作用**：拿着第一步筛选出的“有 EOS 的句子”，把它们在 `eos_idx` 里的保底值替换成**真正的第一个 EOS 的位置索引**。

> 举个例子：
>
> 句子 A 没有 EOS，`eos_idx` 保持保底值 10。
>
> 句子 B 在第 5 个位置出现了 EOS，`eos_idx` 里的值就被更新为 5。

------

#### 2.4.4 张量广播，生成二维掩码

```Python
# 构建有效 Mask：在 EOS 之前的位置为 1，之后填充的 Pad 为 0
completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B, R]
```

广播（Broadcasting）操作，用于将刚才找出的一维长度向量，还原成二维的掩码矩阵。

1. **`torch.arange(is_eos.size(1))`**：生成一个从 $0$ 到 $R-1$ 的基础标尺（比如 `[0, 1, 2, 3, 4]`）。
2. **`.expand(is_eos.size(0), -1)`**：把这根标尺往下复制 `B` 行，变成一个二维矩阵，每一行都是 `[0, 1, 2, 3, 4]`。
3. **`eos_idx.unsqueeze(1)`**：把刚才存有截止位置的一维向量 `[B]` 竖起来，变成 `[B, 1]` 的形状。
4. **`<=` 判断**：用每一行的标尺，去和竖起来的截止位置作比较。
   - 如果当前位置 $\le$ 截止位置，说明属于模型有效的发言，标记为 `True`（后续强转为整数 `1`）。
   - 如果当前位置 $>$ 截止位置，说明是 EOS 后面的废话或填充符，标记为 `False`（转为整数 `0`）。

例子：

假设最大长度是 5，句子 A 的有效长度是 5（没触发 EOS），句子 B 的有效长度是 2（在索引 2 的位置触发了 EOS）。

- **基础标尺矩阵**：

  `[[0, 1, 2, 3, 4],`

  ` [0, 1, 2, 3, 4]]`

- **截止位置竖向量 (`eos_idx`)**：

  `[[4],`

  ` [2]]`

- **执行 `<=` 比较后的 `completion_mask`**：

  `[[1, 1, 1, 1, 1],`  <- 句子 A 全文有效

  ` [1, 1, 1, 0, 0]]`  <- 句子 B 后面全被屏蔽

在后续计算 Loss 时（`per_token_loss * completion_mask`），所有无效位置的 Loss 乘以 0 就被彻底抹除了，这就保证了强化学习的梯度完完全全是干净的。

---

### 2.5 计算 Loss 与反向传播

```python
        # 1. 计算当前模型和参考模型在每个 Token 上的 KL 散度
        kl_div = ref_per_token_logps - per_token_logps  # [B, R]
        
        # 使用精确的离散 KL 估计公式：exp(KL) - KL - 1 (这比简单的平方差更准确)
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B, R]
        
        # 2. 核心策略梯度 Loss (Policy Gradient)
        # 注意 per_token_logps 前面有负号，因为我们要最小化 Loss（即最大化概率 * 优势）
        # 加上 beta * KL 惩罚，防止模型产生奖励黑客行为
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]
        
        # 3. 应用掩码，计算有效序列的平均 Loss
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 4. 加入 MoE 辅助损失并进行梯度累积缩放
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        
        # 反向传播，计算梯度
        loss.backward()

        # 5. [关键] 在反向传播后，调用 tracker 更新历史基线
        response_masks = completion_mask.float()  # [B, R]
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)
```

#### 2.5.1 计算 KL 散度

```Python
        # 1. 计算当前模型和参考模型在每个 Token 上的 KL 散度
        kl_div = ref_per_token_logps - per_token_logps  # [B, R]
        
        # 使用离散 KL 估计公式：exp(KL) - KL - 1 (这比简单的平方差更准确)
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B, R]
```

- **为什么要算 KL？** 强化学习极其容易产生Reward Hacking。如果系统发现满篇说“好好好”能得高分，模型就会迅速退化成只会复读“好好好”的傻子。为了防止模型为拿高分而丧失正常的语言能力，我们需要一个被冻结参数的“参考模型（Ref Model）”作为基准锚点。
- **`kl_div`**：这是参考概率和当前概率的对数差。
- **`torch.exp(kl_div) - kl_div - 1`**：这是一个**正定 KL 估计器**（Schulman 估计器）。
  - 如果模型当前的概率和参考模型完全一样，`kl_div` 是 0。代入公式：$e^0 - 0 - 1 = 0$。模型没有偏离，不惩罚。
  - 只要模型偏离了参考模型（无论概率是变大还是变小），这个公式算出来的值**永远大于 0**，构成了一个“惩罚项”。它比直接用绝对值或平方差在数学上更贴合真实的概率分布差异。

------

#### 2.5.2 构建核心 Loss

```Python
        # 2. 核心策略梯度 Loss (Policy Gradient)
        # 注意 per_token_logps 前面有负号，因为我们要最小化 Loss（即最大化概率 * 优势）
        # 加上 beta * KL 惩罚，防止模型产生奖励黑客行为
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]
```

我们将它拆成两半看：

- **第一部分：`-per_token_logps \* advantages` (策略梯度)**
  - `advantages.unsqueeze(1)` 把刚才算出来的句子级别的 Advantage（形状 `[B]`）拉伸到了 Token 级别（形状 `[B, 1]`），这使得这句话里的每一个字，都共享这句话的总优势得分。
- **第二部分：`+ args.beta \* per_token_kl` (KL 惩罚)**
  - `args.beta` 通常是一个很小的值（如 0.02），它是惩罚力度系数。
  - **总逻辑**：如果模型瞎编了一段偏离常理的话拿了高分，前半部分会想推高概率（奖励），但后半部分的 KL 值会迅速飙升产生巨大的 Loss（惩罚）。两者一综合，模型就会学到：“我要在**不大幅偏离人类正常说话习惯**的前提下，尽量拿高分”。

------

#### 2.5.3 应用掩码与降维求均值

```Python
        # 3. 应用掩码，计算有效序列的平均 Loss
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

刚才算出的 `per_token_loss` 是一个 `[B, R]` 的矩阵，包含了所有句子所有位置的 Loss。

- **`\* completion_mask`**：把我们上一步找出来的Pad的 Loss 直接乘 0 。
- **`.sum(dim=1) / completion_mask.sum(dim=1)`**：对每一句话计算**有效 Token 的平均 Loss**。为什么不直接 `.mean()`？因为直接求均值会把乘了 0 的 Pad 也算作分母，导致短句子的 Loss 被严重稀释。这种写法是变长序列求均值的标准操作。
- **最终 `.mean()`**：把这一个 Batch 里 `B` 个句子的 Loss 再求个平均，最终坍缩成一个**标量（Scalar）** `policy_loss`。

------

#### 2.5.4 梯度累积与反向传播

```Python
        # 4. 加入 MoE 辅助损失并进行梯度累积缩放
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        
        # 反向传播，计算梯度
        loss.backward()
```

- **`+ aux_loss`**：如果你用的是混合专家模型（MoE），必须加上负载均衡损失，否则所有数据都会涌向同一个“专家”导致模型崩溃。
- **`/ args.accumulation_steps`**：这是为了应对显存不足的技巧（梯度累积）。如果我们设置累积 4 步再更新一次参数，那么每一步算出来的 Loss 必须除以 4。这样连续 4 次 `.backward()` 累加起来的梯度，在数学上才严格等于把 Batch Size 扩大 4 倍算出来的梯度。

------

#### 2.5.5 更新价值追踪器

```Python
        # 5. [关键] 在反向传播后，调用 tracker 更新历史基线
        response_masks = completion_mask.float()  # [B, R]
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)
```

- 模型这一步的训练已经完成了，但我们别忘了还要“记账”。
- **`.detach()` 极其关键**：我们把当前模型输出的概率传给 Tracker，去算这一步的 KL 散度（判断策略偏移量）。但是我们**不能**带着梯度传给它。
- 调用 `value_tracker.update`，它就会根据最新的得分和概率变化率，计算出新的 $\alpha$ 和 $\beta$。等下一个 Batch 开始训练时，它就能提供最新的 Baseline 了。

### 2.6 收尾工作

在前面的代码中，我们已经算出了 Loss 并且执行了反向传播。这最后一段代码主要负责四件事：**更新模型权重**、**打印和记录训练日志**、**保存模型权重（存档）**，以及**清理显存垃圾**。

#### 2.6.1 梯度累积与参数更新

```python
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

- **梯度累积 (`accumulation_steps`)**：

  大模型极其占用显存，我们单卡可能一次只能塞下 `Batch_Size = 2` 的数据。但为了让训练更稳定，我们希望相当于 `Batch_Size = 8` 的效果。怎么办？ 做法是：计算完 Loss 并 `backward()` 之后，**先不更新参数**，而是让梯度在显存里累加（之前 Loss 除以了 `accumulation_steps` 就是为了等比例缩小每次的梯度）。当循环达到 `accumulation_steps`（比如 4 次）后，才执行一次真正的参数更新。

- **梯度裁剪 (`clip_grad_norm_`)**： 强化学习的梯度非常容易“爆炸”（偶尔遇到极端惩罚或奖励）。这里设置了一道安全门：如果所有参数梯度的平方和（L2范数）超过了设定的阈值（`args.grad_clip`，比如 1.0），就按比例把它们强行缩小。这能防止模型“一步迈太大”扯断了原本学好的知识。

------

#### 2.6.2 数据监控与日志上报

```python
        if step % args.log_interval == 0 or step == iters:
            # 提取标量数值 (.item()) 供打印
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            # ... 其他指标提取省略 ...
```

- **`loss.item() \* args.accumulation_steps`**：因为前面算 Loss 的时候除以了累积步数，这里为了在日志里显示**真实**的 Loss，又给乘了回来。

```python
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '...)

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    # ...
                })
```

------

#### 2.6.3 保存模型

```python
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            # ... 路径和名字处理 ...
            
            # 剥洋葱第一层：剥离多卡分布式包装器 (DDP)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 剥洋葱第二层：剥离 torch.compile 的底层加速包装器
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            
            state_dict = raw_model.state_dict()
            # 转为半精度 (FP16) 并移至 CPU 保存，大幅节省硬盘和内存
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 保存断点续训需要的全套装备（包括优化器状态等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, ...)
            
            model.train()
            del state_dict
```

- **“脱壳”是工程细节的重点**：在训练时，模型被包上了多卡并行的外壳（DDP）甚至还包了编译加速的外壳（`torch.compile`）。如果直接存，以后推理时加载就会报错说“找不到 `module.xxxx` 参数”。所以必须用 `.module` 和 `_orig_mod` 把真正纯粹的模型参数抠出来存。
- **存成 `.half().cpu()`**：神经网络在显存里训练时可能用了占用空间极大的高精度类型（FP32），存盘时转成半精度（FP16/BF16）直接让文件大小缩减一半。
- 这里做了双重保险：`torch.save` 存了一个干净的权重供以后推理用；`lm_checkpoint` 把优化器里的动量状态、当前跑到第几步等信息也存了下来，供下次“断点续训”用。

------

#### 2.6.4 显存管理

```python
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines, response_masks
```

- **为什么要手动 `del`？** 在 PyTorch 中，只要一个张量（Tensor）的变量名还在作用域里，它占用的显存就不会被释放。在强化学习中，我们要同时在显存里塞下 3 个大模型（Actor, Ref, Reward），每一轮还要生成极其庞大的计算图和各种中间矩阵（比如 `per_token_logps`、`advantages` 等）。