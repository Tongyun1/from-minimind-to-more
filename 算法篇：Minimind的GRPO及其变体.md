# 算法篇：Minimind的GRPO及其变体

## 1 引言

长期以来，PPO（Proximal Policy Optimization）作为大模型强化学习（RLHF）的基石算法，统治了整个对齐阶段。然而，随着 Agentic 和 Reasoning 模型（如 DeepSeek-R1、OpenAI o1）的崛起，传统的 PPO 由于依赖庞大且难以训练的 Critic 模型，在极高的显存开销和“稀疏奖励评估难”的问题上面临巨大瓶颈。近期，**GRPO（Group Relative Policy Optimization）** 横空出世，彻底移除了 Critic 模型，通过同 Prompt 下的组内相对得分评估优势（Advantage），不仅大幅降低了训练成本，更在数学推理和代码生成等客观评判任务中展现出惊人的潜力。

尽管如此，原生 GRPO 并非银弹。在长思维链（Long CoT）和混合专家（MoE）架构中，它暴露出长度惩罚偏误、方差爆炸、探索能力衰退等局限性。为了突破这些瓶颈，学术界与工业界衍生出了一系列进阶算法：

- **Dr. GRPO**：从数学底层修正了基线与长度偏差，防止模型“靠凑字数作弊”；
- **DAPO**：通过解耦裁剪释放了长文本的探索潜力，缓解策略坍塌；
- **GSPO**：将重要性采样提升至序列级，稳住了 MoE 架构极易崩溃的训练方差；
- **SAPO**：用温度控制的软门控机制榨干了每一次采样的梯度价值；
- **GTPO**：利用策略熵实现了无过程奖励模型下的精细化信用分配。
- 此外，**PRM（过程奖励模型）** 与 **STaR（自学推理者）** 等机制的引入，更是补齐了复杂推理数据冷启动与步骤级验证的短板。

本文会先讲清楚Minimind的GRPO实现，然后再讲讲其他变体。**这是今年大模型算法岗必考的核心知识点**，请不要跳过。

## 2 数据准备：`lm_dataset.py`

GRPO的数据集和PPO完全一样，都是用的`lm_dataset.py`中的RLAIFDataset类。详情请参考《算法篇：Minimind的PPO》。这里就不多讲了。

---

## 3 GRPO流程讲解

这里先简要讲讲GRPO的大致流程

### 3.1 GRPO 的核心输入与设置

在 GRPO 开始运行之前，系统需要配置好以下核心组件和超参数：

- **输入提示词 (Prompt/Query, $q$)**：来自训练数据集的用户问题或指令。
- **策略模型 (Actor Model, $\pi_\theta$)**：当前正在训练的大语言模型，负责生成回答。
- **参考模型 (Reference Model, $\pi_{\text{ref}}$)**：策略模型在强化学习前的快照（通常是 SFT 阶段的模型），其参数在训练过程中被冻结，用于限制策略模型的更新幅度，防止模型“遗忘”原有能力。
- **奖励模型 (Reward Model, $RM$) 或规则系统**：用于对生成的回答进行打分。在代码或数学场景下，这也可以是一个基于规则的校验器（Rule-based Verifier）。
- **采样组大小 ($G$)**：一个超参数，表示针对同一个提示词 $q$，策略模型需要生成的独立回答数量。

---

### 3.2 GRPO 的训练流程

GRPO 的核心思想是通过“组内比较”来确定哪些回答更好，而不是依赖一个全局的绝对基准。类似于推荐系统中的列表级排序（Listwise Ranking），相对优劣比绝对分值更指导模型的进化。

1. **群体采样 (Group Sampling)**：

   给定一个提示词 $q$，当前的策略模型 $\pi_\theta$ 会生成 $G$ 个不同的输出（回答），记为 $\{o_1, o_2, \dots, o_G\}$。

2. **奖励计算 (Reward Computation)**：

   奖励模型或规则验证器对这 $G$ 个输出分别进行评估，得到对应的绝对奖励分数 $\{r_1, r_2, \dots, r_G\}$。

3. **计算相对优势 (Advantage Estimation)**：

   对这组绝对奖励进行标准化处理（Z-score normalization），计算出每个输出的相对优势 $A_i$。公式如下，其中 $\mu$ 和 $\sigma$ 分别是该组奖励的均值和标准差：

   $$A_i = \frac{r_i - \mu}{\sigma}$$

   优势 $A_i > 0$ 表示该回答在同组中表现高于平均水平，应当被鼓励；反之则应当被抑制。

4. **计算 KL 散度惩罚 (KL Divergence Penalty)**：

   为了防止策略模型 $\pi_\theta$ 偏离参考模型 $\pi_{\text{ref}}$ 太远，GRPO 在每个生成的 token 级别计算直接的 KL 散度估计，并将其作为惩罚项加入。

5. **策略更新 (Policy Update)**：

   模型通过最大化以下目标函数来更新参数 $\theta$（结合了截断机制以保证训练稳定性）：

   $$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$

---

### 3.3 流程图解

可以将 GRPO 的单次迭代流程拆解为以下逻辑流：

1. **[环境]** 提供 Question $\rightarrow$ **[Actor 模型]**。
2. **[Actor 模型]** 并行生成 Output 1, Output 2 ... Output $G$。
3. **[环境 / 奖励模型]** 对所有 Output 独立打分，输出 Reward 1, Reward 2 ... Reward $G$。
4. **[计算模块]** 汇总这 $G$ 个 Reward，求均值与方差，将数值转化为正负交错的 Advantage (优势值)。
5. **[Actor 模型]** 与 **[Reference 模型]** 对比输出概率，计算 KL 惩罚。
6. **[优化器]** 结合 Advantage 和 KL 惩罚，计算梯度并更新 **[Actor 模型]** 的权重。

------

### 3.4 GRPO vs PPO 流程的主要区别

传统 RLHF 中广泛使用的 PPO (Proximal Policy Optimization) 依赖于一个额外的 Critic（价值模型）来预测绝对 baseline。GRPO 则巧妙地剔除了这个庞大的组件。

| **比较维度**             | **PPO (Proximal Policy Optimization)**                       | **GRPO (Group Relative Policy Optimization)**                |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **所需模型数量**         | 4个：Actor, Reference, Reward, **Critic**                    | 3个：Actor, Reference, Reward                                |
| **基线预测 (Baseline)**  | 依赖 Critic 模型预测输入状态的绝对价值 (Value) 作为基准。    | 依赖组内同级输出的均值作为相对基准。                         |
| **显存占用**             | **极高**。Critic 模型通常与 Actor 模型同等参数量，需占用大量显存。 | **大幅降低**。直接砍掉了 Critic 模型，节省了整整一个 LLM 的显存开销。 |
| **优势函数 (Advantage)** | 基于广义优势估计 (GAE)，计算单次输出与 Critic 预测值之差。   | 基于群体均值方差归一化，通过同批次 $G$ 个样本的内部竞争计算。 |
| **计算效率**             | 每次迭代需要做 Actor 和 Critic 的前向和反向传播。            | 只需做 Actor 的前向和反向传播，推理和训练速度更快。          |

## 4 训练代码主体`train_ppo.py`

### 4.1 calculate_rewards函数

这个函数的主要职责是：**接收模型生成的回复，并计算出一个综合得分（Reward张量），告诉策略模型这次回答得有多好。**

它工作流程拆解为两个主要阶段：**规则奖励（Rule-based Reward）** 和 **模型打分奖励（Model-based Reward）**。

#### 1. 阶段一：规则/启发式奖励 (仅在 `args.reasoning == 1` 时触发)

如果当前训练的是一个“推理模型”（类似于 DeepSeek-R1 这种需要先思考再回答的模型），函数会通过内部的 `reasoning_model_reward` 进行严格的格式审查。它包含两部分得分：

- **严格格式分 (0.5分)：** 使用正则表达式检查回复是否**完全**符合 `<think>思考过程</think>\n<answer>最终答案</answer>` 的排版。只要稍微错位或者多出其它无关文本，这 0.5 分就直接丢掉。
- **标签完整分 (最高1.0分)：** 使用 `mark_num` 统计 `<think>`, `</think>`, `<answer>`, `</answer>` 这四个关键标签的数量。每个标签出现且仅出现一次，给 0.25 分。这是一种“软性”引导，即使格式不完美，只要模型学会输出这些关键标记，也能拿到保底分。

#### 2. 阶段二：外部奖励模型打分 (Reward Model Score)

这部分是所有模式下都会执行的核心打分逻辑。它不依赖死板的规则，而是用另一个预训练好的 AI（即传入的 `reward_model`）来评估回答的内容质量。

- **组装对话上下文：** 函数会遍历所有的 `prompts` 和模型生成的 `responses`。通过正则提取出用户的问题，并把模型刚刚生成的回复作为 `assistant` 拼接到对话列表中（标准的 ChatML 格式）。
- **模型评估：** 调用 `reward_model.get_score` 给这段对话打分。这个分数反映了回答的逻辑性、相关性和安全性等综合素质。
- **截断保护 (Clipping)：** `score = max(min(score, scale), -scale)`。为了防止奖励模型偶尔“发疯”给出极其夸张的过高或过低分数（这会导致强化学习梯度爆炸，模型崩溃），代码把分数强行限制在 $[-3.0, 3.0]$ 的区间内。

#### 3. 特殊机制：对“推理答案”的加强打分

如果你开启了推理模式（`args.reasoning == 1`），代码在这个阶段还有一个非常巧妙的设计：

- 它会用正则表达式单独把 `<answer>` 和 `</answer>` 中间的**最终答案**提取出来。
- 让奖励模型**只针对这个最终答案**再打一次分（`answer_score`）。
- **加权融合：** `score = score * 0.4 + answer_score * 0.6`。这意味着，整个包含废话和思考过程的回复只占 40% 的权重，而**最终结论的正确与否占据了 60% 的主导地位**。这能逼迫模型把注意力放在“得出正确结论”上，而不是写一堆看似合理但毫无用处的思考过程。

这个函数最终返回的 `rewards` 是一个张量（Tensor），形状为[B*num_gen]，num_gen为每个prompt生成的样本数，里面的每一个数值都是：**格式得分 + 标签得分 + (外部模型对全句打分 \* 0.4 + 外部模型对答案打分 \* 0.6)**。这个分数随后会被送入 GRPO 算法中，用于计算优势（Advantage），从而指导模型更新参数。

代码还是比较好懂的，直接看注释吧。

```python
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    # prompts: list[str], length B
    # responses: list[str], length B*num_gen
    
    def reasoning_model_reward(rewards):
        """基于回答格式和标签规则的启发式奖励函数 (Rule-based Reward)"""
        # 正则表达式：严格匹配以 <think> 开始和结束，接着以 <answer> 开始和结束的格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # 兼容 <think> 和 <answer> 之间有一个空行的格式
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        
        # 检查每个生成的 response 是否完全符合上述两种格式之一
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            # 如果格式完全匹配，给予 0.5 的基础格式奖励
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        # 将格式奖励叠加到总奖励张量上
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            """统计关键标签的出现次数，每出现一次正确标签给予额外奖励"""
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        # 计算批次中每个 response 的标签数量奖励
        mark_rewards = [mark_num(response) for response in responses]
        # 将标签奖励叠加到总奖励张量上
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化一个全 0 的张量，用于存储每个 response 的最终得分
    rewards = torch.zeros(len(responses), device=args.device)
    
    # 如果启用了推理模型模式（args.reasoning == 1），则先计算格式相关的启发式奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards) # 形状为[B*num_gen]

    # 禁用梯度计算，因为这部分只用于推理外部奖励模型
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0 # 设置奖励模型的截断范围为 [-3.0, 3.0]

        # 遍历批次中的每个 prompt
        for i in range(batch_size):
            # 遍历针对该 prompt 生成的 num_generations 个不同的 response
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 解析 prompt，将其从纯文本转换为标准的 ChatML / 消息列表格式
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 将模型生成的 response 作为 assistant 角色追加到消息列表中
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                # 调用外部奖励模型评估整个对话，得到一个打分
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                # 将分数限制在设定好的 scale 范围内，防止极端值导致训练崩溃
                score = max(min(score, scale), -scale)

                # 如果是推理模型，还需要对 <answer> 标签内的最终答案进行单独评分
                if args.reasoning == 1:
                    # 提取 <answer> 和 </answer> 之间的内容
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        # 构建仅包含最终答案的对话进行评分
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        # 综合得分：整个回复得分占 40%，核心答案得分占 60%
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        # 将列表转换为张量并加到总 rewards 上
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards # 形状为[B*num_gen]，num_gen为每个prompt生成的样本数
```

---

### 4.2 grpo_train_epoch函数，用于训练

#### 1. 数据准备与 Prompt Token化

训练开始，首先从 DataLoader 中取出一个批次（Batch）的用户提示词（Prompts）。

```python
prompts = batch['prompt']  # list[str], 长度为 Batch Size (B)
# 将文本转化为模型能看懂的 Token ID
prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                          padding_side="left", add_special_tokens=False).to(args.device)

# 如果设置了最大长度，则从左侧截断（保留最新内容，为右侧生成做准备）
if args.max_seq_len:
    prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
    prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

**关键点：** 注意这里使用的是 `padding_side="left"`。因为在做因果语言模型（Causal LM）的自回归生成时，所有的生成都是接在最右侧的，所以左侧填充能保证右侧是对齐的。这里也与上一篇文章中提到的数据集类RLAIFDataset中没有token化相对应，在这里才进行token化。

#### 2. 策略模型生成回复 (Rollout 阶段)

模型根据用户的 Prompt 自由发挥，生成对应的回答。

```python
with torch.no_grad(): # 纯生成阶段，不需要计算梯度，节省大量显存
  	# DDP 模型需要使用 .module 访问 generate 方法
    model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model 
    # 让模型为每个 prompt 生成多个回答 (num_return_sequences=args.num_generations)
    outputs = model_for_gen.generate(
        **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
        num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id) # [B*num_gen, P+R]，P是Prompt长度，R是生成的响应长度

# outputs 包含了 prompt + 生成的回答。我们把生成的回答部分单独切出来
completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # 形状: # # [B*num_gen, R]，R是生成的响应长度
```

**关键点：** 此时，对于 `B` 个问题，模型已经生成了 `B * num_generations` 个不同的回答。

#### 3. 计算对数概率 (Log Probabilities)

这是强化学习的基础：我们需要知道模型生成当前这些字的“自信程度”（概率）。

```python
# 一个闭包小函数：给定 token 序列，计算模型生成这些 token 的对数概率
def get_per_token_logps(mdl, input_ids, n_keep):
    # input_ids: [B*num_gen, P+R]，P是Prompt长度，R是生成的响应长度
    # n_keep: int, 保留的token数量
    # return: torch.Tensor, 形状为[B*num_gen, R]
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids # 如果input_ids是推理模式，则需要detach克隆
    logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :] # 形状为[B*num_gen, P+R, V]，V是词汇表大小
    per_token_logps = [] # 形状为[B*num_gen, R]
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row # 如果ids_row是推理模式，则需要detach克隆
        per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)) # 形状为[B*num_gen, R]
    return torch.stack(per_token_logps) # 形状为[B*num_gen, R]

with autocast_ctx:
    # 1. 计算当前策略模型生成这些字的概率 (开启梯度计算，这是我们要优化的对象！)
    per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
    
with torch.no_grad():
    # 2. 计算参考模型 (未经微调的老模型) 生成一模一样字的概率 (不计算梯度，作为锚点)
    ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))
```

大模型在第二步“生成回复”时，是通过采样（Sampling）随机吐出文字的。但在反向传播更新参数时，我们不能用“随机采样的字”去求导，而是必须精确算出模型生成这每一个字的**真实数学概率**。

下面是 `get_per_token_logps` 详细解释：

1. **核心概念：为什么要算对数概率 (Log Probability)？**

大模型的本质是文字接龙。给定前面的词 $x_{<t}$，模型会输出词表中所有词作为下一个词 $x_t$ 的概率分布 $P(x_t | x_{<t})$。

由于连续相乘的概率（比如 $0.1 \times 0.2 \times 0.05 \dots$）很快就会变得极小导致计算机浮点数下溢，所以我们通常对概率取自然对数（Log）。将乘法变成加法：

$$\log P(X) = \sum_{t} \log P(x_t | x_{<t})$$

2. **代码拆解：`get_per_token_logps` 函数**

这个函数的目标是：**传入一个完整的句子（Prompt + 回答），提取出其中“回答”部分每一个 token 的对数概率。**

**第一步：获取模型原始输出 (Logits) 并进行自回归错位**

```python
# mdl: 传入的模型 (策略模型或参考模型)
# input_ids: 完整的序列 (Prompt + 回答)
# n_keep: 我们只需要保留最后 n_keep 个 token 的概率 (即生成的回答部分)

# 拿取模型的原始输出 Logits
logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
```

**详细解释：**

- **Logits：** 这是模型最后一层输出的、还没有经过 Softmax 归一化的原始得分矩阵。它的形状是 `[Batch, Seq_Len, Vocab_Size]`。
- **错位切片 `[:, :-1, :]`：** 这是一个极其关键的自回归操作！因为语言模型是用第 $t$ 个位置的特征去预测第 $t+1$ 个位置的词。所以我们要把 Logits 的最后一个位置丢掉（因为它预测的是序列外未知的下一个词），这样切片后的 Logits 矩阵正好和我们要评估的目标 token 序列在位置上一一对应了。

第二步：提取真实生成 Token 的对应概率 (Gather 操作)

```python
per_token_logps = []
# 遍历 Batch 中的每一行
# logits_row 形状: [Seq_Len, Vocab_Size]
# ids_row 是模型实际生成的 Token ID，截取最后 n_keep 个
for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
    # 1. 先把 Logits 转为对数概率
    log_probs = logits_row.log_softmax(dim=-1) 
    
    # 2. 从巨大的词表概率中，精准“抠”出实际生成的那个字的概率
    gathered_logps = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
    
    per_token_logps.append(gathered_logps)
```

**详细解释 `torch.gather` 到底在干嘛：**

假设词表大小是 10 万。`log_probs` 是一个巨大的矩阵，每一行有 10 万个浮点数（代表预测下一个字是词表中每个字的概率）。

但我们不关心这 10 万个字，我们**只关心模型实际上生成的那个字**！

比如模型在这一步实际上生成了词表索引为 `345` 的词（假设是“苹”字）。`torch.gather` 的作用就是：拿着 `ids_row` 里的 ID（比如 `345`），去 `log_probs` 对应的行里，把第 `345` 列的那个数值提取出来。

这样，我们就得到了**模型真实生成的这句回答中，每一个字的生成概率**。

**第三步：打包返回**

```python
return torch.stack(per_token_logps) # 形状变为 [B*num_gen, R]
```

把列表中所有的张量堆叠起来，变成一个整齐的矩阵。

**3. 外层调用：策略模型 vs 参考模型**

回到 `grpo_train_epoch` 的主体，你会看到这个函数被调用了两次：

```python
with autocast_ctx:
    # 1. 计算当前“策略模型”的对数概率
    # 注意：这里没有 torch.no_grad()，所以这个 per_token_logps 是带有梯度计算图的！
    # 它是我们后续用强化学习公式进行梯度下降求导的源头。
    per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))

with torch.no_grad():
    # 2. 计算冻结的“参考模型”的对数概率
    # 这个模型完全不参与训练，它的输出只是一个固定的参考数值（锚点）。
    ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))
```

**为什么同样的东西要算两遍？**

因为我们需要防止模型为了拿高分而“走火入魔”（比如发现奖励模型喜欢感叹号，就通篇输出感叹号）。

我们算出 `ref_per_token_logps`（老模型原本是怎么说话的），再算出 `per_token_logps`（新模型现在是怎么说话的），两者相减就可以计算出 **KL 散度 (KL Divergence)**。KL 散度越大约等于惩罚越重，从而强迫新模型在提高分数的同时，依然保持老模型原本的语言逻辑和流利度。

---

#### 4. 奖励结算与相对优势 (Advantage) 计算 ： GRPO 的灵魂

这就是我们前面讨论过的“分组内卷”逻辑。

```python
# 把 token ID 解码回文本
completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True) # list[str], length B*num_gen
# 用之前讲解的 calculate_rewards 裁判函数，给所有的回答打分
rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device) # [B*num_gen]

# === GRPO 组内优势计算 ===
grouped_rewards = rewards.view(-1, args.num_generations)  # 按问题分组 [B, num_gen]
# 组内平均分和标准差
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations) # [B*num_gen]
std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations) # [B*num_gen]

# Advantage = (奖励 - 组内均值) / 组内标准差
advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10) # 截断防爆炸 # [B*num_gen]
```

**关键点：** `advantages` 如果是正数，说明这个回答在同组中表现拔尖；如果是负数，说明拖了后腿。

#### 5. 计算 KL 散度与 Padding Mask

为了防止模型“学聪明了”，只学会钻奖励模型的空子而忘记了怎么正常说话，我们需要用 KL 散度把它拉住，让它别偏离参考模型太远。

```python
# 计算结束符 (EOS) 后的 Mask，忽略掉填充的无意义字符
is_eos = completion_ids == tokenizer.eos_token_id
# ... (计算出 completion_mask，有效字符位置为1，无效为0)

# 计算参考模型与当前模型的 KL 散度
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
```

#### 6. 计算最终 Loss 并反向传播

万事俱备，终于到了计算损失函数的时刻。我们要最大化优秀回答的概率，同时最小化与参考模型的偏离。

```python
# 核心 PPO/GRPO Loss 公式！
# torch.exp(per_token_logps - per_token_logps.detach()) 相当于重要性采样比率 (ratio)
# 乘以优势 (advantages)，再减去 KL惩罚 (args.beta * per_token_kl)
per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)

# 利用 Mask 把无效 token 的损失过滤掉，求序列的平均 Loss
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
loss = (policy_loss + aux_loss) / args.accumulation_steps  # 加上 MoE 的辅助 Loss (如果有)

# 反向传播求梯度
loss.backward()
```

#### 7. 优化器步进与日志更新

最后，根据累积的梯度更新参数。

```python
if (step + 1) % args.accumulation_steps == 0:
    if args.grad_clip > 0:
        # 梯度裁剪，防止偶尔出现的巨大梯度把模型毁了
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()  # 真正更新模型参数
    scheduler.step()  # 学习率衰减
    optimizer.zero_grad() # 清除这一步的梯度，为下一步做准备

# 打印日志、记录到 Wandb 并在达到固定步数时保存 Checkpoint...
```

**总结：** `grpo_train_epoch` 就像是一个严谨的流水线：**出题 (Prompt) -> 答题 (Rollout) -> 打分 (Reward) -> 组内排名 (Advantage) -> 分析得失 (Loss & KL) -> 自我反省并进步 (Backward & Step)**。

---

### 4.3 if __name__ == "__main__"部分

这一部分没啥大的改动，就不讲了

## 5 关于GRPO的一些讨论

### 5.1 GRPO 的核心流程与“连坐”机制

GRPO 的基础流程非常直接：**多生几个 -> 算算平均分 -> 谁在平均分之上就学谁**。

这种方法极其适合“结果导向”的任务（如数学、编程）。因为只要结果对了，我们就可以通过对比，自动筛选出那些导致正确结果的推理步骤（Chain of Thought），而不需要一个极其聪明（且昂贵）的 Critic 模型来一步步指导。

#### 5.1.1 优势值（Advantage）的整句统一

在标准的 GRPO 实现（如 DeepSeek-R1）中，优势（Advantage）通常不是逐 Token 变化的，而是**整句（Sequence-level）统一**的。也就是说，对于同一条回复中的第 1 个 Token 和第 100 个 Token，它们被分配的优势值是完全一样的。

#### 5.2.2 信用分配 (Credit Assignment) 难题

这听起来似乎比 PPO “粗糙”，但它之所以能工作，背后有其统计学原理。你可能会问：如果优势是一样的，模型怎么知道是哪一步推理做对了？

这就涉及到了强化学习中的经典难题。假设我写了 100 行代码，只错了一个变量名导致运行失败：

- **PPO 的理想情况**：Critic 能精准指出第 99 行那个变量名有问题（前提是 Critic 足够强）。
- **GRPO 的连坐机制**：会给这 100 行代码全部打低分（负优势）。那模型岂不是把前面 99 行正确的逻辑也“冤枉”了？

#### 5.1.3 GRPO 的解法：采样与统计平均

GRPO 的解决方案是：**依靠“采样”和“统计平均”**。想象一下，针对同一个问题，GRPO 采样了 8 组回答（Group Size = 8）：

- **回答 1 (失败)**：前面逻辑对，第 99 步错了 $\rightarrow$ **全体负分**
- **回答 2 (成功)**：前面逻辑对，第 99 步也对 $\rightarrow$ **全体正分**
- **回答 3 (失败)**：第一步就错了 $\rightarrow$ **全体负分**

当模型进行梯度更新时：

1. **对于“前面的正确逻辑”**：它在“回答 1”中被惩罚，但在“回答 2”中被奖励。如果采样的样本够多，只要它是导致成功的必要条件，它总会更有可能出现在高分样本中。平均下来，它的概率会被推高。
2. **对于“第 99 步的错误”**：它主要出现在负分样本中，所以会被抑制。

**结论：** GRPO 虽然单次看是“连坐”（一人犯错，全句受罚），但通过大量数据的统计，模型最终能学会区分“哪些 Token 是真正导致成功的关键”。

------

### 5.2 GRPO 的 Loss 函数与梯度细节

我们看一下 GRPO 的 Loss 函数背后的逻辑：

注意这里的优势值 $A$ 对于该序列中的所有 Token 都是常数。但是，对数概率 $\log \pi_\theta(a_t|s_t)$ 的梯度是针对每个 Token 单独计算的。

这意味着：**虽然信号强弱（$A$）是一样的，但每个 Token 对梯度的贡献取决于它当前的概率分布。**

虽然 DeepSeek-R1 的标准用法是 Outcome-based（结果导向，整句优势），但 GRPO 理论上也支持逐 Token 的优势，前提是你有一个能提供逐 Token 奖励的机制：

- **Process Reward (过程奖励)**：如果你有一个外部脚本或模型，能给每一个步骤（Step）打分。
- **KL 惩罚**：KL 散度天然是逐 Token 的。

在某些实现中，为了稳定训练，会将总优势拆解为：

$$A_t = A_{\text{seq}} - \beta \text{KL}_t$$

在这种情况下，因为引入了逐 Token 的 KL 惩罚，最终用于更新的优势 $A_t$ 实际上在每个 Token 上也是微小变化的。但总体来说，主导 GRPO 梯度的核心信号（也就是那个 $A_{\text{seq}}$）依然是整句级别的。

------

### 为什么 Agentic RL 这么爱 GRPO？

在 DeepSeek-R1 等强推理模型发布之前，PPO 是绝对的主流。但在 Agentic/Reasoning 场景下，GRPO 正在迅速取代 PPO，同时它与 DPO 有着本质的适用场景区别。

#### 5.3.1 GRPO vs. PPO：为了“去肥增瘦”与“摆脱 Critic”

在 Agentic RL 中（如让模型写代码、解复杂数学题），PPO 存在两个巨大的痛点，而 GRPO 完美解决了它们：

**优势一：无需 Critic 模型（极致的显存效率）**

- **PPO 的痛点**：标准 PPO 算法需要维护 4 个模型（Actor, Reference, Reward Model, Critic）。在训练超大模型（如 70B+）时，Critic 模型本身也非常大，导致显存占用极高，训练成本主要消耗在“陪跑”的 Critic 上。
- **GRPO 的解法**：直接抛弃了 Critic 模型。它不通过神经网络来预测 Value，而是通过 Group（一组采样）的平均奖励作为基准（Baseline）。
- **收益**：显存占用大幅降低，计算资源可以全部集中在优化 Actor 上。你可以用同样的资源训练更大的模型，或者支持更长的 Context（这对 Agentic 推理至关重要）。

**优势二：避开了“价值估计难”的问题**

- **PPO 的痛点**：在长程推理（CoT）或 Agent 任务中，训练一个好的 Critic 非常难。比如模型写了 50 行代码，Critic 很难准确判断“第 20 行代码对最终跑通有没有帮助”。Critic 估值不准，PPO 训练就会震荡甚至崩塌。
- **GRPO 的解法**：使用 **Group Relative（组内相对）** 比较。模型生成 8 个答案，GRPO 只需判断相对好坏，这比让神经网络去绝对预测“这一步值多少分”要准确和稳定得多。

#### 5.3.2 GRPO vs. DPO：为了“探索”与“无中生有”

DPO 在 RLHF（对齐）中非常流行，但在 Agentic RL 中，GRPO 具有不可替代的优势：

**优势一：Online Exploration（在线探索） vs. Offline Data（离线数据）**

- **DPO (Offline)**：需要预先准备好成对的数据（好回答 vs 坏回答）。模型只是在学习“模仿好的，远离坏的”，无法让模型学会它没见过的东西。
- **GRPO (Online)**：典型的在线强化学习。模型在训练过程中不断尝试（Sample），一旦它偶然做对了一次（Aha Moment），奖励函数就会给予高分，模型就会强化这条路径。
- **收益**：GRPO 具备“泛化”和“涌现”能力。能激发模型通过试错，找到人类数据集中没有覆盖到的解题路径。

**优势二：处理“部分正确”的能力**

- **DPO**：通常处理的是由人类标注的 Preference（偏好）。
- **GRPO**：非常适合结合 Rule-based Reward（规则奖励）。在代码场景中，我们可以定义详细规则（通过编译 0.2 分，测试用例过半 0.5 分，全对 1.0 分）。GRPO 会在一组生成中，自动分析哪些特征导致了更高分数，精细化优化策略。

------

### 5.4 总结

如果把 Agentic RL 比作培养一个解题高手：

1. **PPO 像是请了一个昂贵的私教 (Critic)**：每做一步都给你打分。但这私教太贵（显存贵），而且在复杂难题上，私教也经常看走眼（Value 估计不准）。
2. **DPO 像是背题库**：看着标准答案背诵，确实能学会规范，但遇到新题（Out-of-distribution）就懵了，缺乏独立思考能力。
3. **GRPO 像是搞题海战术的小组学习**：不要私教（省显存）。针对一道题，让脑子里的不同想法“打架”（采样一组）。谁最后做对的测试用例多，谁就是老大（Group Relative Baseline），下次就按它的思路想。

**结论**：在需要长思维链（Long CoT）、客观真值（Ground Truth）验证、以及希望模型涌现出超越数据能力的 Agentic RL 任务中，GRPO 相对于 PPO 的显存优势和相对于 DPO 的探索优势，使其成为当前最“香”的选择。

## 6 Dr. GRPO：修正 GRPO 的内在优化偏差 (GRPO Done Right)

**面试考点：为什么标准 GRPO 会导致模型越回答越长（尤其是答错的时候）？如何修正？**

虽然 GRPO 在内存和计算效率上取得了巨大成功，但近期前沿研究（如对 R1-Zero 训练过程的剖析）指出，标准的 GRPO 在目标函数设计上存在细微的数学理论偏误，这会导致模型产生不良的“作弊”行为。

### 6.1 痛点分析：标准 GRPO 的数学偏误

标准 GRPO 存在两个隐蔽的优化偏差（Optimization Bias），导致了模型**长度惩罚/奖励的错位**：

1. **基线偏差 (Baseline Bias)**： 标准 GRPO 在计算组内相对优势 $A_i$ 时，使用该组的均值作为 Baseline，但在底层梯度推导时，其对应的缩放因子（Scaling Factor）常被设定为 $\frac{1}{K}$（$K$ 为组内样本数）。从策略梯度定理（Policy Gradient Theorem）的无偏估计角度来看，严格的数学推导表明应该使用 $\frac{1}{K-1}$ 来进行无偏校正。
2. **响应级长度偏差 (Response-level Length Bias)**： 在很多标准的代码实现中（包括之前提到的 Sequence-level Loss 的计算），通常会将整条序列的总 Loss 除以该序列生成的 Token 数量 $|o|$ 来求平均。这个看似常规的操作在 GRPO 中带来了致命的副作用：
   - **对于正确的答案（优势** $A > 0$**）**：除以长度 $|o|$ 会让模型在优化梯度时觉得“用更少的字拿到同样的正向优势”更划算，这在无意中**过度惩罚了正确思维链的长度**，抑制了模型进行长程探索和深度思考的能力。
   - **对于错误的答案（优势** $A < 0$**）**：除以长度 $|o|$ 会严重稀释负面惩罚！模型很快会发现一个漏洞：“只要我回答错误，我就尽可能瞎扯得很长，这样每个 Token 分摊到的惩罚系数就变小了”。这就是为什么很多使用原生 GRPO 训练的模型在遇到难题时，会陷入“无限复读”或输出极长且无意义废话的根本原因。

### 6.2 Dr. GRPO 的核心改进

Dr. GRPO（GRPO Done Right）旨在通过严格推导修正这些理论上的偏误，提出了一种无偏的策略优化方法：

- **无偏优势估计**：使用正确的统计学基线和缩放因子重构 Advantage 的计算公式。
- **剔除不合理的长度除法**：彻底修正了简单粗暴地将总体 Loss 除以响应长度 $|o|$ 的操作，确保无论是长还是短，每个 Token 对模型更新的惩罚或奖励力度都是公平且对齐的。

**结果**：Dr. GRPO 成功消除了模型“靠凑字数来稀释错误惩罚”的作弊行为。通过根除长度偏误，它在显著提高 Token 训练效率（Token Efficiency）和节约算力成本的同时，使模型在复杂推理任务上的上限表现更加稳定。

## 7 DAPO：Decoupled Clip and Dynamic sAmpling Policy Optimization

### 7.1 痛点分析

GRPO 的核心亮点是**移除了 Critic（价值模型）**。对于一个 prompt，它用当前策略生成 $G$ 个回答，然后在组内计算相对优势（Advantage）：

$$A_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

这极大地节省了显存和推理计算量。但在实际进行 Long-CoT（长思维链）训练时，暴露出以下几个致命痛点：

**1. 优势为零时的算力浪费（无效采样）**

如果在这一组 $G$ 个回答中，模型全对或者全错（即所有 $R_i$ 都相等），此时组内的标准差 $\text{std}$ 为 0。经过处理后，所有样本的优势 $A_i$ 都变成了 0。这意味着这些样本对模型更新**完全没有贡献**（梯度消失），白白浪费了大量昂贵的生成（Rollout）算力。

**2. 熵坍塌与探索能力受限（对称裁剪问题）**

像 PPO 一样，GRPO 限制了新旧策略概率比值 $r_t$ 的变化范围，通常是严格对称的 $[1-\epsilon, 1+\epsilon]$（例如 $[0.8, 1.2]$）。但在长推理中，如果某个极其关键的“好 Token”在旧策略下的概率非常低，由于上限被死死卡在 $1.2$，模型对这个好方向的鼓励被过早截断（Capped early）。这会导致模型倾向于只输出安全、重复的废话，系统多样性丧失，发生**熵坍塌（Entropy Collapse）**。

**3. 长度偏置（样本级 Loss 的天然缺陷）**

原生的 GRPO 计算 Loss 是在**样本级别（Sample-level）**。它先计算每个回复序列内部所有 Token 的平均 Loss，然后再把多个样本的 Loss 平均。

这就导致了一个严重问题：一条包含 1000 个 Token 的高质量长推理，和一条只有 10 个 Token 的短回复，在全局梯度更新时的权重是一样的。这变相惩罚了模型去进行复杂的长推理。

**4. 截断带来的奖励噪声（Reward Noise）**

训练时为了防止 OOM，通常会设置最大生成长度。如果模型还没输出完就被强行截断，规则奖励系统（Rule-based RM）通常会直接给低分。但模型无法区分“我是因为逻辑错了被扣分”还是“我是因为话没说完被扣分”，这引入了巨大的噪声。

### 7.2 DAPO 的核心改进

**1.Dynamic Sampling（动态采样）—— 解决算力浪费**

- 在生成数据后，DAPO 会动态检查这一组数据的奖励标准差。如果 $\text{std} = 0$（全对或全错，没有相对优势），直接**丢弃（Skip）**这组数据，不参与计算。
- 模型会持续采样，直到缓存中积累了足够多 $\text{std} > 0$（有有效梯度）的样本，才拼成一个完整的 Batch 进行网络更新。
- **改进效果：** 确保每一次反向传播都在做有效更新，极大提升了训练效率和收敛速度。

**2.Clip-Higher（解耦的非对称裁剪）—— 解决熵坍塌**

既然好 Token 的概率上升空间被限制，DAPO 提出了**非对称裁剪（Asymmetric Clipping）**，解耦了上下限。

- 它将鼓励侧（Upper bound）的上限放宽，比如从 $1.2$ 提高到 $1.28$。
- 抑制侧（Lower bound）的下限保持不变（依然是 $0.8$）。
- **改进效果：** 既能防止坏 Token 的概率无底线暴跌导致模型崩溃，又能给低概率的优质 Token 更大的上升空间，有效维持了输出的多样性。

**3.Token-Level Policy Gradient Loss（Token 级损失）—— 解决长度偏置**

- DAPO 将 GRPO 的样本级归一化改成了 **Token 级归一化**。
- 它不再先求单条样本的均值，而是把 Batch 内所有样本的所有 Token 拉平，直接在这个巨大的 Token 集合上计算策略梯度 Loss。
- **改进效果：** 长度越长、思考越深入的高质量回答，在 Loss 中占据的权重就越大，从底层逻辑上鼓励模型在必要时进行 Long-CoT 推理。

**4.Overlong Filtering & Shaping（超长过滤与软惩罚）—— 解决截断噪声**

- **过滤（Filtering）：** 如果一条数据是因为达到最大长度被截断的，DAPO 会在计算 Loss 时直接 Mask 掉它，避免模型学到错误的截断逻辑。
- **软惩罚（Soft Overlong Punishment）：** 为了防止模型为了水字数而无限循环（比如像死循环一样的无意义重复思考），DAPO 引入了长度感知惩罚。如果回复过长，会在原本的奖励基础上扣除一个随长度增长的惩罚值。

## 8 GSPO：序列级组相对策略优化 (Group Sequence Policy Optimization)

### 8.1 痛点分析

虽然 GRPO 去掉了 Critic 模型，极大降低了显存开销，但在面对超长逻辑推理（Long-CoT）和超大规模的混合专家模型（MoE）时，它暴露出一个底层设计上的致命缺陷：**Token 级的优化与 Sequence 级的奖励不匹配**。

具体来说，有以下几个痛点：

**1. Token 级重要性采样带来的“高方差与梯度爆炸”** GRPO 在计算新旧策略的差异时，是计算每一个 Token 的概率比（Importance Ratio）。但在强化学习中，一个 Token 在一次生成中只被采样一次。在长达几千字的长思考序列中，个别极其生僻或概率波动极大的 Token 会导致整个重要性采样的乘积剧烈震荡。这种高方差的噪声很容易导致梯度不稳定，甚至训练直接崩溃。

**2. 奖励与优化的粒度错位（Mismatch）** 我们在训练推理模型时，奖励（Reward）通常是给整个句子的（比如：这道数学题最终做对了得 1 分，做错了得 0 分）。这是一个 **Sequence-level（序列级）** 的信号。但是，GRPO 却把这个宏观的奖励，强行分配给每一个 Token，并在 **Token-level（Token 级）** 上进行裁剪和优化。

**3. MoE 架构下的路由崩溃（Routing Drift）** 在训练大规模 MoE 模型（如 DeepSeek-V3 或 Qwen 系列）时，由于 GRPO Token 级的梯度噪声太大，会导致 MoE 的路由网络（Router）在每次更新后发生剧烈偏移。为了防止专家负载不均衡或模型崩盘，GRPO 往往需要引入非常复杂且极其消耗算力的工程 Hack 手段，比如 **Routing Replay（路由重放）**，这让训练成本再次飙升。

**4. 对工程框架的精度极度敏感** 因为 GRPO 优化到 Token 级别，训练引擎（如 Megatron）和推理引擎（如 vLLM）在底层浮点数计算上的微小精度差异，都会在长序列中被无限放大，导致新旧策略概率比（Logprob ratio）计算失准。

### 8.2 GSPO 的核心改进

**1.序列级概率比与长度归一化（Sequence-Level Likelihood Ratio）**

GSPO 不再挨个计算 Token 的概率比，而是直接计算**整个回答序列**在新旧策略下的似然比。

为了防止长序列导致这个比值呈指数级爆炸或趋于零，GSPO 极其巧妙地引入了**长度归一化（取几何平均）**。其重要性权重 $s_i(\theta)$ 的公式为：

$$s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}}$$

这里 $|y_i|$ 是序列的长度。

- **改进效果：** 彻底消除了单个 Token 带来的剧烈方差噪声，让重要性采样变得极其平滑和稳定。

**2.序列级裁剪（Sequence-Level Clipping）**

和 PPO/GRPO 裁剪单个 Token 的概率比不同，GSPO 的目标函数直接对上述的**序列级权重 $s_i(\theta)$** 进行裁剪（例如限制在 $[1-\epsilon, 1+\epsilon]$ 之间），然后再乘以序列级的优势（Advantage）$A_i$。

- **改进效果：** 完美对齐了“序列级奖励”和“序列级优化”。模型不再因为某几个异常 Token 而被过度惩罚或鼓励，整体逻辑链条的连贯性得到了更好的保护。

**3.原生稳定的 MoE 训练（抛弃 Routing Replay）**

得益于序列级更新带来的极低方差和极高稳定性，GSPO 的梯度信号变得非常清晰。

- **改进效果：** Qwen 团队发现，使用 GSPO 训练庞大的 MoE 模型时，路由网络不再发生剧烈漂移。因此，可以**完全废弃掉昂贵的 Routing Replay** 机制，模型依然能稳定收敛，大幅提升了训练吞吐量。

#### 

## 9 SAPO：平滑软优势策略优化 (Soft Advantage Policy Optimization)

SAPO 是由阿里 Qwen 团队近期提出的一项重磅改进（目前已集成在 `ms-swift` 和 Hugging Face 的 `trl` 框架中）。既然你之前深入研究过 **MoE（混合专家模型）** 的底层架构，理解 SAPO 会非常自然，因为 SAPO 解决的核心痛点之一，恰恰是在给带有复杂路由的 MoE 模型做 RL 时极易引发的崩溃问题。

### 9.1 痛点分析：硬截断的资源浪费

**1. 学习信号的丢失（Zero Gradients）**

如果在长文本生成中，某个极具创造性的 Token 概率翻倍了（比如 $r_t = 2.0$），超出了 $1.2$ 的上限，GRPO 会直接把它的梯度抹零。这意味着模型不仅没有被鼓励，反而白白浪费了一次宝贵的探索。在动辄几千 Token 的推理链中，大量的有效梯度因为“出界”而被直接丢弃，导致样本效率极低。

**2. MoE 架构下剧烈震荡**

在包含海量专家的 MoE 模型中，新旧模型哪怕只发生了一点点路由变化（Routing Heterogeneity），同一个 Token 的输出概率也可能产生天壤之别，导致概率比 $r_t$ 极不稳定：

- **如果裁剪区间调得太窄（紧）：** 大量 Token 会疯狂触发边界，梯度全是 0，模型根本学不到东西。
- **如果裁剪区间调得太宽（松）：** 极其离谱的 Off-policy 噪声 Token 就会混进来，直接把梯度引爆，导致训练瞬间崩溃（NaN）。

(注：GSPO 为了解决这个问题，把粒度提升到了“序列级”。但 GSPO 也有副作用——如果一个几千字的优秀推理序列里只混进了几个极其离谱的错误 Token，GSPO 会把整条序列的梯度都压制掉，这属于“连坐惩罚”。)

### 9.2 SAPO 的核心改进

**1.温度控制的软门控机制（Soft Gate Function）**

SAPO 用一个基于 Sigmoid 的**连续动态门控函数**替换了 GRPO 那个带棱角的 Clip 截断。

当概率比 $r_t$ 开始偏离安全区时，SAPO 不会突然把梯度变成 0，而是让梯度呈现一种**平滑衰减（Smooth Decay）**。

- **改进效果：** 彻底消除了梯度断崖式下跌的问题。偏离得越多，给的更新权重越小，但**永远不会一刀切变成 0**。这既防止了异常 Token 搞崩模型，又尽可能多地榨取了每一个采样数据的学习信号（极大地提升了样本利用率）。

**2.非对称温度调节（Asymmetric Temperature）**

这是 SAPO 最精妙的细节设计。在 RL 训练中，“鼓励好行为”和“惩罚坏行为”的风险是不一样的：

- 当优势为正（$A > 0$）时，我们是在**提高**某个正确 Token 的概率，这相对安全。
- 当优势为负（$A < 0$）时，我们是在**压低**某个错误 Token 的概率。但在庞大的词表中，压低一个 Token，往往意味着其他成千上万个垃圾 Token 的 Logits 会被动上升，这极其容易引发模型输出乱码或退化。

因此，SAPO 为正负优势引入了不同的温度系数（$\tau_{pos}$ 和 $\tau_{neg}$）：

它赋予了负优势**更高的温度（更陡峭的衰减）**。也就是说，在惩罚坏 Token 时，SAPO 的态度更加保守和谨慎，防止因为惩罚力度过大而把其他无关的词表概率搞崩。

**3.完美兼容 Token 级与现有基建（Drop-in Replacement）**

不同于 GSPO 需要把整个重要性采样拉高到序列级别，SAPO 依然保持在 **Token 级别**操作。

- **改进效果：** 它可以像插件一样，无缝替换掉现有 GRPO/PPO 代码库里计算 Loss 的那几行代码。不需要更改奖励归一化逻辑，不需要改动优势函数的计算方式。

## 10 GTPO：组 Token 级策略优化 (Group Token Policy Optimization)

**面试考点：没有过程奖励（PRM），如何在极长的思考过程中进行精准的信用分配？**

### 10.1 痛点分析

GRPO 本质上是基于结果的（Outcome-based），即常说的 ORM（Outcome Reward Model）。这就导致了一个“连坐”问题：模型写了 1000 字的思维链（CoT），最后答案蒙对了，GRPO 就会把这 1000 个字统一赋予正向 Advantage。但实际上，这 1000 字里可能包含了一段完全错误的逻辑。这就是经典的**稀疏信用分配（Sparse Credit Assignment）**难题。

### 10.2 GTPO 的核心改进：动态熵权重

**1.将“策略熵（Policy Entropy）”作为重要性探针**

GTPO 提出了一个极其聪明且直觉的假设：**在正确的推理序列中，模型表现出高熵（高不确定性、在多个选项中纠结）的位置，往往就是推理链条中最关键的“决策点（Decision Points）”或认知努力最大的地方。**

相反，像标点符号、固定句式的生成，模型的确定性极高（熵极低）。

**2.Token 级的动态奖励再分配**

在计算出当前序列的全局优势 $A_i$ 后，GTPO 不再把它平均分给所有 Token。相反，它会提取模型在生成每个 Token 时的内部熵 $H_{i,t}$，并计算出一个动态权重：

$$w_{i,t} \propto \frac{H_{i,t}}{\sum_{k} H_{k,t}}$$

接着，把原始的序列级优势 $A_i$ 乘以这个权重，得到一个真正属于该 Token 的细粒度优势：

$$A_{i,t} = A_i \times w_{i,t}$$

- **改进效果：** 彻底打破了平均主义！如果模型最终答对了题，那么在生成过程中那些让模型“绞尽脑汁、高度不确定”的关键 Token，会分到最大比例的奖励；而那些水到渠成、闭着眼睛都能生成的低熵 Token，只分到很小的奖励。

**3.负向序列的防崩溃机制**

对于回答**错误**的序列（负面奖励），GTPO 的处理非常谨慎。因为错误可能是由某一个致命的“愚蠢决定”导致的，但高熵并不一定代表那个致命错误发生的位置。因此，在处理负向 Advantage 时，GTPO 通常会回退到更平缓的分配方式，或者结合我们上文提到的 SAPO 类似的软截断，防止误伤无辜 Token。

**4.“白嫖”的伪过程奖励**

- **改进效果：** GTPO 最惊艳的一点在于，策略熵（Logits 的分布情况）是模型在 Forward（前向传播）生成文本时**天然就会计算出来的副产物**。GTPO 巧妙地“白嫖”了这个内部信号，完全不需要引入外部的 PRM 网络，就实现了类似过程奖励的效果，极大地提升了样本利用率和上限（Ceiling）。



**总结表**

| **算法名称**                                    | **想解决什么核心痛点 **                                      | **核心改进机制 **                                            | **Advantage 计算粒度**                 | **Loss 优化粒度**                |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------- | -------------------------------- |
| **PPO**  *(Proximal Policy)*                    | **(行业基石)** 传统策略梯度更新易崩溃；但在大模型时代，**极其消耗显存**（需同时跑 Actor, Critic, Ref, Reward 四个模型）。 | **Actor-Critic 架构 + GAE + 硬裁剪**：引入专门的价值网络（Critic）逐字评估，限制新旧策略概率比的更新幅度。 | **Token 级**  (基于 Critic 的状态价值) | **Token 级**  (硬裁剪)           |
| **GRPO**  *(Group Relative)*                    | PPO 的 Critic 模型太吃显存，导致无法在消费级硬件或超大模型上高效训练。 | **组内相对优势打分**：彻底移除 Critic，通过同一 Prompt 生成的一组完整回答的均值/标准差计算优势。 | **序列级**  (Sequence)                 | **Token 级**  (宏观分数平均分配) |
| **DAPO**  *(Decoupled Clip & Dynamic Sampling)* | GRPO 优势为 0 时浪费算力；硬裁剪限制了长推理中优质低概率词的探索。 | **动态采样 + 解耦裁剪 + 全局 Loss**：过滤无效样本，单向放宽正向鼓励的上限，底层逻辑倾向于长序列。 | **序列级**  (外加动态过滤)             | **Token 级**  (全局 Token 拉平)  |
| **GSPO**  *(Group Sequence)*                    | GRPO 单 Token 计算方差过大、极其不稳定；引发 MoE 模型路由发生灾难性漂移。 | **序列级似然比 + 长度归一化**：放弃单 Token 概率比，直接计算整条序列的新旧概率比并做几何平均裁剪。 | **序列级**  (Sequence)                 | **序列级**  (Sequence Clip)      |
| **SAPO**  *(Soft Adaptive)*                     | PPO/GRPO 的硬裁剪（Hard Clip）一刀切导致梯度频繁归零；惩罚错误时容易带崩模型。 | **软门控 + 非对称温度系数**：用平滑衰减代替生硬截断，对“惩罚（负优势）”设置更谨慎的温度系数。 | **序列级**  (Sequence)                 | **Token 级**  (引入软门控)       |
| **GTPO**  *(Group Token)*                       | GRPO 宏观奖金平均分给所有 Token（大锅饭），无法区分“划水词”和“关键决策”。 | **动态熵权重重塑**：白嫖模型的内部策略熵，模型越纠结（高熵）的关键决策点，分到的 Advantage 越多。 | **Token 级**  (按熵值动态再分配)       | **Token 级**  (加权更新)         |





## 11 PRM：过程奖励模型 (Process Reward Model)

**面试考点：为什么解数学题和写代码，PRM 比 ORM 更重要？**

虽然前文提到的算法都在试图弥补只看结果的缺陷，但在攻克极度复杂的数学定理证明或大型工程代码时，**PRM（过程奖励模型）** 依然是不可逾越的护城河（如 DeepSeek-Math 的成功就高度依赖 PRM）。

### 11.1 ORM vs PRM

- **ORM (Outcome Reward Model)**：只看最终结果。优点是数据好获取（比如代码是否通过测试用例），缺点是反馈极其稀疏，模型不知道中间哪一步走错了。
- **PRM (Process Reward Model)**：对模型推理的**每一步（Step-by-Step）**进行独立打分。总奖励 $R = \sum r_{\text{step}}$。

### 11.2 PRM 的价值与挑战

在基于 PPO 或 GRPO 的架构中挂载 PRM 后，模型在生成推理轨迹时，可以获得密集的正负反馈。如果第 3 步算错了，PRM 立即给负分，后续生成的优势值 A 就会被切断，逼迫模型学习正确的中间逻辑。 **难点**：PRM 的标注成本极其高昂。目前主流的做法是结合**蒙特卡洛树搜索（MCTS）**自动生成大量的逻辑分支，或者利用基于规则的验证器（代码编译器、符号学工具）来自动化构建 PRM 的训练数据。

## 12 STaR：自学推理者 (Self-Taught Reasoner)

**面试考点：什么是大模型的“左脚踩右脚”起飞？（推理数据的冷启动机制）**

在聊完 RL 算法后，必须了解一个前置概念：**STaR**。它虽然不是严格意义上的 RL 策略梯度算法，但它是目前所有推理模型（包括 OpenAI o1, DeepSeek-R1 早期冷启动）生成高质量训练数据的核心思想。

### 12.1 STaR 的运行逻辑

假设我们只有问题和最终答案（只有题干和选项），没有中间的推导过程。STaR 提出了一个极具优雅的 Bootstrapping（自举）循环：

1. **生成 (Generate)**：让当前语言模型针对问题生成思维链（Rationale）和答案。
2. **过滤 (Filter)**：比对最终答案。把做对的那些样本（连同它的思维链）直接加入到微调数据集中。
3. **合理化补充 (Rationalization)**：对于做错的问题，把**正确的答案直接告诉模型**（作为 Hint），命令模型：“答案是 X，请你倒推并写出为什么是 X 的思维链”。如果这次推导逻辑通顺，也将其加入数据集。
4. **微调 (Fine-tune)**：使用这批自己生成的、包含正确逻辑的高质量数据对模型进行 SFT（监督微调）。
5. **循环**：拿着变聪明的模型，重新回到第 1 步。

### 12.2 STaR 在 RL 体系中的地位

强化学习（GRPO/PPO）需要模型本身具备一定的基础概率去命中正确答案，否则就会陷入“永远得不到正奖励”的死循环。STaR 通过“生成-过滤-反思”机制，用极低的成本为大模型注入了初始的 CoT（思维链）能力，为后续接入 GRPO 算法进行无止境的上限探索铺平了道路。









**到这里大模型的强化学习基础就差不多了，看完这些你应该能应对大多数八股问题。但是如果真的要入门大模型强化学习科研，还是要自己跑一些代码。**

