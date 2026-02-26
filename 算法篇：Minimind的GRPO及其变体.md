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

   $$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q,\{o_i\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,\ \operatorname{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\mathrm{KL}}\!\left(\pi_\theta \| \pi_{\text{ref}}\right) \right) \right]
$$


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

### if __name__ == "__main__"部分

这一部分没啥大的改动，就不讲了

## 4.3 关于GRPO的一些讨论

### 一、 GRPO 的核心流程与“连坐”机制

GRPO 的基础流程非常直接：**多生几个 -> 算算平均分 -> 谁在平均分之上就学谁**。

这种方法极其适合“结果导向”的任务（如数学、编程）。因为只要结果对了，我们就可以通过对比，自动筛选出那些导致正确结果的推理步骤（Chain of Thought），而不需要一个极其聪明（且昂贵）的 Critic 模型来一步步指导。

#### 1. 优势值（Advantage）的整句统一

在标准的 GRPO 实现（如 DeepSeek-R1）中，优势（Advantage）通常不是逐 Token 变化的，而是**整句（Sequence-level）统一**的。也就是说，对于同一条回复中的第 1 个 Token 和第 100 个 Token，它们被分配的优势值是完全一样的。

#### 2. 信用分配 (Credit Assignment) 难题

这听起来似乎比 PPO “粗糙”，但它之所以能工作，背后有其统计学原理。你可能会问：如果优势是一样的，模型怎么知道是哪一步推理做对了？

这就涉及到了强化学习中的经典难题。假设我写了 100 行代码，只错了一个变量名导致运行失败：

- **PPO 的理想情况**：Critic 能精准指出第 99 行那个变量名有问题（前提是 Critic 足够强）。
- **GRPO 的连坐机制**：会给这 100 行代码全部打低分（负优势）。那模型岂不是把前面 99 行正确的逻辑也“冤枉”了？

#### 3. GRPO 的解法：采样与统计平均

GRPO 的解决方案是：**依靠“采样”和“统计平均”**。想象一下，针对同一个问题，GRPO 采样了 8 组回答（Group Size = 8）：

- **回答 1 (失败)**：前面逻辑对，第 99 步错了 $\rightarrow$ **全体负分**
- **回答 2 (成功)**：前面逻辑对，第 99 步也对 $\rightarrow$ **全体正分**
- **回答 3 (失败)**：第一步就错了 $\rightarrow$ **全体负分**

当模型进行梯度更新时：

1. **对于“前面的正确逻辑”**：它在“回答 1”中被惩罚，但在“回答 2”中被奖励。如果采样的样本够多，只要它是导致成功的必要条件，它总会更有可能出现在高分样本中。平均下来，它的概率会被推高。
2. **对于“第 99 步的错误”**：它主要出现在负分样本中，所以会被抑制。

**结论：** GRPO 虽然单次看是“连坐”（一人犯错，全句受罚），但通过大量数据的统计，模型最终能学会区分“哪些 Token 是真正导致成功的关键”。

------

### 二、 GRPO 的 Loss 函数与梯度细节

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

### 三、 为什么 Agentic RL 这么爱 GRPO？

在 DeepSeek-R1 等强推理模型发布之前，PPO 是绝对的主流。但在 Agentic/Reasoning 场景下，GRPO 正在迅速取代 PPO，同时它与 DPO 有着本质的适用场景区别。

#### 1. GRPO vs. PPO：为了“去肥增瘦”与“摆脱 Critic”

在 Agentic RL 中（如让模型写代码、解复杂数学题），PPO 存在两个巨大的痛点，而 GRPO 完美解决了它们：

**优势一：无需 Critic 模型（极致的显存效率）**

- **PPO 的痛点**：标准 PPO 算法需要维护 4 个模型（Actor, Reference, Reward Model, Critic）。在训练超大模型（如 70B+）时，Critic 模型本身也非常大，导致显存占用极高，训练成本主要消耗在“陪跑”的 Critic 上。
- **GRPO 的解法**：直接抛弃了 Critic 模型。它不通过神经网络来预测 Value，而是通过 Group（一组采样）的平均奖励作为基准（Baseline）。
- **收益**：显存占用大幅降低，计算资源可以全部集中在优化 Actor 上。你可以用同样的资源训练更大的模型，或者支持更长的 Context（这对 Agentic 推理至关重要）。

**优势二：避开了“价值估计难”的问题**

- **PPO 的痛点**：在长程推理（CoT）或 Agent 任务中，训练一个好的 Critic 非常难。比如模型写了 50 行代码，Critic 很难准确判断“第 20 行代码对最终跑通有没有帮助”。Critic 估值不准，PPO 训练就会震荡甚至崩塌。
- **GRPO 的解法**：使用 **Group Relative（组内相对）** 比较。模型生成 8 个答案，GRPO 只需判断相对好坏，这比让神经网络去绝对预测“这一步值多少分”要准确和稳定得多。

#### 2. GRPO vs. DPO：为了“探索”与“无中生有”

DPO 在 RLHF（对齐）中非常流行，但在 Agentic RL 中，GRPO 具有不可替代的优势：

**优势一：Online Exploration（在线探索） vs. Offline Data（离线数据）**

- **DPO (Offline)**：需要预先准备好成对的数据（好回答 vs 坏回答）。模型只是在学习“模仿好的，远离坏的”，无法让模型学会它没见过的东西。
- **GRPO (Online)**：典型的在线强化学习。模型在训练过程中不断尝试（Sample），一旦它偶然做对了一次（Aha Moment），奖励函数就会给予高分，模型就会强化这条路径。
- **收益**：GRPO 具备“泛化”和“涌现”能力。能激发模型通过试错，找到人类数据集中没有覆盖到的解题路径。

**优势二：处理“部分正确”的能力**

- **DPO**：通常处理的是由人类标注的 Preference（偏好）。
- **GRPO**：非常适合结合 Rule-based Reward（规则奖励）。在代码场景中，我们可以定义详细规则（通过编译 0.2 分，测试用例过半 0.5 分，全对 1.0 分）。GRPO 会在一组生成中，自动分析哪些特征导致了更高分数，精细化优化策略。

------

### 四、 总结

如果把 Agentic RL 比作培养一个解题高手：

1. **PPO 像是请了一个昂贵的私教 (Critic)**：每做一步都给你打分。但这私教太贵（显存贵），而且在复杂难题上，私教也经常看走眼（Value 估计不准）。
2. **DPO 像是背题库**：看着标准答案背诵，确实能学会规范，但遇到新题（Out-of-distribution）就懵了，缺乏独立思考能力。
3. **GRPO 像是搞题海战术的小组学习**：不要私教（省显存）。针对一道题，让脑子里的不同想法“打架”（采样一组）。谁最后做对的测试用例多，谁就是老大（Group Relative Baseline），下次就按它的思路想。

**结论**：在需要长思维链（Long CoT）、客观真值（Ground Truth）验证、以及希望模型涌现出超越数据能力的 Agentic RL 任务中，GRPO 相对于 PPO 的显存优势和相对于 DPO 的探索优势，使其成为当前最“香”的选择。

## 5 Dr. GRPO：修正 GRPO 的内在优化偏差 (GRPO Done Right)

**面试考点：为什么标准 GRPO 会导致模型越回答越长（尤其是答错的时候）？如何修正？**

虽然 GRPO 在内存和计算效率上取得了巨大成功，但近期前沿研究（如对 R1-Zero 训练过程的剖析）指出，标准的 GRPO 在目标函数设计上存在细微的数学理论偏误，这会导致模型产生不良的“作弊”行为。

### 5.1 痛点分析：标准 GRPO 的数学偏误

标准 GRPO 存在两个隐蔽的优化偏差（Optimization Bias），导致了模型**长度惩罚/奖励的错位**：

1. **基线偏差 (Baseline Bias)**： 标准 GRPO 在计算组内相对优势 $A_i$ 时，使用该组的均值作为 Baseline，但在底层梯度推导时，其对应的缩放因子（Scaling Factor）常被设定为 $\frac{1}{K}$（$K$ 为组内样本数）。从策略梯度定理（Policy Gradient Theorem）的无偏估计角度来看，严格的数学推导表明应该使用 $\frac{1}{K-1}$ 来进行无偏校正。
2. **响应级长度偏差 (Response-level Length Bias)**： 在很多标准的代码实现中（包括之前提到的 Sequence-level Loss 的计算），通常会将整条序列的总 Loss 除以该序列生成的 Token 数量 $|o|$ 来求平均。这个看似常规的操作在 GRPO 中带来了致命的副作用：
   - **对于正确的答案（优势** $A > 0$**）**：除以长度 $|o|$ 会让模型在优化梯度时觉得“用更少的字拿到同样的正向优势”更划算，这在无意中**过度惩罚了正确思维链的长度**，抑制了模型进行长程探索和深度思考的能力。
   - **对于错误的答案（优势** $A < 0$**）**：除以长度 $|o|$ 会严重稀释负面惩罚！模型很快会发现一个漏洞：“只要我回答错误，我就尽可能瞎扯得很长，这样每个 Token 分摊到的惩罚系数就变小了”。这就是为什么很多使用原生 GRPO 训练的模型在遇到难题时，会陷入“无限复读”或输出极长且无意义废话的根本原因。

### 5.2 Dr. GRPO 的核心改进

Dr. GRPO（GRPO Done Right）旨在通过严格推导修正这些理论上的偏误，提出了一种无偏的策略优化方法：

- **无偏优势估计**：使用正确的统计学基线和缩放因子重构 Advantage 的计算公式。
- **剔除不合理的长度除法**：彻底修正了简单粗暴地将总体 Loss 除以响应长度 $|o|$ 的操作，确保无论是长还是短，每个 Token 对模型更新的惩罚或奖励力度都是公平且对齐的。

**结果**：Dr. GRPO 成功消除了模型“靠凑字数来稀释错误惩罚”的作弊行为。通过根除长度偏误，它在显著提高 Token 训练效率（Token Efficiency）和节约算力成本的同时，使模型在复杂推理任务上的上限表现更加稳定。

## 6 DAPO：解耦裁剪与动态采样 (Decoupled Advantage Policy Optimization)

**面试考点：如何解决长文本强化学习中的“策略坍塌”与“探索能力下降”？**

虽然 GRPO 摆脱了 Critic 模型，但在处理极长上下文（如长链条数学推理或复杂代码生成）时，传统 PPO/GRPO 中对称的 $\epsilon$-裁剪机制（通常将更新比率限制在 $[1-\epsilon, 1+\epsilon]$）暴露出了致命缺陷。

### 6.1 痛点分析

在长文本生成中，模型偶尔会“灵光一闪”，在极低的概率下生成了某个非常关键且正确的 Token（此时 $\pi_\theta / \pi_{\theta_{\text{old}}}$ 极大，远超 $1+\epsilon$）。按照标准的 GRPO 硬裁剪逻辑，由于超出了上界，这部分极具探索价值的梯度被直接切断了。这不仅抑制了模型探索罕见正确答案的能力，还会导致策略在次优解上坍塌（Entropy Collapse）。

### 6.2 DAPO 的核心改进

DAPO (Decoupled Advantage Policy Optimization) 的核心思路是**非对称的解耦裁剪（Asymmetric Clip-higher Mechanism）**：

- **放宽正优势上界**：对于表现好（Advantage > 0）的探索性输出，DAPO 显著放宽甚至动态调整上限裁剪的阈值，鼓励模型大口“吃进”这些导致高分的罕见 Token 的概率。
- **严格保留负优势下界**：对于表现差（Advantage < 0）的输出，依然保持严格的下界裁剪，防止模型因为一次失败的尝试而过度调低原本正常的 Token 概率。

此外，DAPO 往往伴随**动态采样机制**，对于方差大、难区分好坏的 Prompt 增加采样组大小 $G$，对于简单的 Prompt 减少采样，从而在不增加总计算量的前提下，极大提升了长文本训练的收敛速度和稳定性。

## 7 GSPO：序列级组相对策略优化 (Group Sequence Policy Optimization)

**面试考点：如何从根本上解决 MoE 模型和长序列训练中的方差爆炸？**

在 DeepSeek 等大厂的实践中，研究人员发现直接将 GRPO 应用于 MoE（混合专家）模型或超长序列时，经常会出现训练极不稳定的情况，甚至需要引入额外的 Routing Replay（路由重放）等补丁技术才能勉强收敛。

### 7.1 痛点分析：Token 级重要性采样的原罪

传统 GRPO 在计算重要性采样比率（Importance Sampling Ratio）时是在 Token 级别进行的：

$$w_{t}^{\text{GRPO}} = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

对于长序列而言，单步微小的概率扰动会在整个序列层面积累。特别是在 MoE 模型中，轻微的概率变化会导致 Token 被分配给完全不同的专家网络，这种 Token 级别的高频噪声带来了极其剧烈的梯度方差。

### 7.2 GSPO 的核心改进

GSPO 将重要性采样的粒度从微观的 Token 级提升到了宏观的 Sequence（序列）级。它通过对整个句子的生成概率比值求几何平均，计算出一个全局平滑的权重：

$$w^{\text{GSPO}} = \left( \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)} \right)^{\frac{1}{|y|}} = \exp \left( \frac{1}{|y|} \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_{t}|y_{<t}, x)}{\pi_{\theta_{\text{old}}}(y_{t}|y_{<t}, x)} \right)$$

**优势**：

1. **方差极大降低**：由于采用了几何平均，个别 Token 的极端概率比率被整个序列的长度摊平，有效过滤了异常噪声。
2. **原生适配 MoE**：极其平滑的梯度信号使得 MoE 模型的门控网络（Router）能够稳定更新，直接废弃了繁琐的 Routing Replay 补丁，成为了训练多模态和复杂推理 MoE 模型的首选。

## 8 SAPO：平滑软优势策略优化 (Soft Advantage Policy Optimization)

**面试考点：如何提高强化学习中正负样本的利用率？**

### 8.1 痛点分析：硬截断的资源浪费

不管是 PPO 还是 GRPO，其裁剪函数 `clip(ratio, 1-eps, 1+eps)` 都是一个“阶跃式”的硬截断。这意味着，一旦某个动作的概率偏离旧模型超过了 20%（假设 $\epsilon=0.2$），它的梯度瞬间变为 0。在昂贵的大模型训练中，这意味着我们辛辛苦苦做了一次前向传播，结果有一大半 Token 的梯度直接被丢弃了，样本利用率极低。

### 8.2 SAPO 的核心改进

SAPO 引入了**温度控制的软门控（Soft Gate）机制**来替代暴力的截断。它通常使用缩放的 Tanh 函数或 Sigmoid 函数来平滑地压制极端概率比率：

$$\mathcal{L}_{\text{SAPO}} = \text{SoftGate}\left( \frac{\pi_\theta}{\pi_{\text{old}}}, \tau \right) \cdot A$$

其中 $\tau$ 是温度系数。 当偏离度较小时，它的表现和普通线性更新一样；当偏离度变大时，它不会瞬间把梯度切断为0，而是像弹簧一样，提供一个逐渐减弱的柔和梯度。 **结果**：SAPO 实现了 Token 级别的自适应梯度平滑，不仅能防止策略更新过猛，还使得几乎每一个生成的 Token 都能为模型参数更新提供微小但有效的贡献，显著拉升了算力利用性价比。

## 9 GTPO：组 Token 级策略优化 (Group Token Policy Optimization)

**面试考点：没有过程奖励（PRM），如何在极长的思考过程中进行精准的信用分配？**

### 9.1 痛点分析

GRPO 本质上是基于结果的（Outcome-based），即常说的 ORM（Outcome Reward Model）。这就导致了一个“连坐”问题：模型写了 1000 字的思维链（CoT），最后答案蒙对了，GRPO 就会把这 1000 个字统一赋予正向 Advantage。但实际上，这 1000 字里可能包含了一段完全错误的逻辑。这就是经典的**稀疏信用分配（Sparse Credit Assignment）**难题。

### 9.2 GTPO 的核心改进：动态熵权重

在没有外部昂贵的过程奖励模型（PRM）的情况下，GTPO 巧妙地利用了模型自身的**策略熵（Policy Entropy）**来进行 Token 级别的奖励塑形（Reward Shaping）。

GTPO 认为：如果在一个关键的决策分叉口，模型当时非常犹豫（输出的概率分布极其平缓，即**熵很高**），但它最终选择的那个 Token 导致了整个序列获得了高分，那么这个“关键决定”就应该拿首功！ 反之，如果在一些毫无悬念的废话环节（如格式符号，熵极低），即使最终结果好，也不应该给它们分配过高的权重。

GTPO 通过将宏观的 Sequence Advantage 与微观的 Token 级别熵值相乘，实现了**免 Critic 的细粒度信用分配**，极大地缓解了长思维链中“逻辑崩坏但答案凑对（Think-Answer Mismatch）”的现象。

## 10 PRM：过程奖励模型 (Process Reward Model)

**面试考点：为什么解数学题和写代码，PRM 比 ORM 更重要？**

虽然前文提到的算法都在试图弥补只看结果的缺陷，但在攻克极度复杂的数学定理证明或大型工程代码时，**PRM（过程奖励模型）** 依然是不可逾越的护城河（如 DeepSeek-Math 的成功就高度依赖 PRM）。

### 10.1 ORM vs PRM

- **ORM (Outcome Reward Model)**：只看最终结果。优点是数据好获取（比如代码是否通过测试用例），缺点是反馈极其稀疏，模型不知道中间哪一步走错了。
- **PRM (Process Reward Model)**：对模型推理的**每一步（Step-by-Step）**进行独立打分。总奖励 $R = \sum r_{\text{step}}$。

### 10.2 PRM 的价值与挑战

在基于 PPO 或 GRPO 的架构中挂载 PRM 后，模型在生成推理轨迹时，可以获得密集的正负反馈。如果第 3 步算错了，PRM 立即给负分，后续生成的优势值 A 就会被切断，逼迫模型学习正确的中间逻辑。 **难点**：PRM 的标注成本极其高昂。目前主流的做法是结合**蒙特卡洛树搜索（MCTS）**自动生成大量的逻辑分支，或者利用基于规则的验证器（代码编译器、符号学工具）来自动化构建 PRM 的训练数据。

## 11 STaR：自学推理者 (Self-Taught Reasoner)

**面试考点：什么是大模型的“左脚踩右脚”起飞？（推理数据的冷启动机制）**

在聊完 RL 算法后，必须了解一个前置概念：**STaR**。它虽然不是严格意义上的 RL 策略梯度算法，但它是目前所有推理模型（包括 OpenAI o1, DeepSeek-R1 早期冷启动）生成高质量训练数据的核心思想。

### 11.1 STaR 的运行逻辑

假设我们只有问题和最终答案（只有题干和选项），没有中间的推导过程。STaR 提出了一个极具优雅的 Bootstrapping（自举）循环：

1. **生成 (Generate)**：让当前语言模型针对问题生成思维链（Rationale）和答案。
2. **过滤 (Filter)**：比对最终答案。把做对的那些样本（连同它的思维链）直接加入到微调数据集中。
3. **合理化补充 (Rationalization)**：对于做错的问题，把**正确的答案直接告诉模型**（作为 Hint），命令模型：“答案是 X，请你倒推并写出为什么是 X 的思维链”。如果这次推导逻辑通顺，也将其加入数据集。
4. **微调 (Fine-tune)**：使用这批自己生成的、包含正确逻辑的高质量数据对模型进行 SFT（监督微调）。
5. **循环**：拿着变聪明的模型，重新回到第 1 步。

### 11.2 STaR 在 RL 体系中的地位

强化学习（GRPO/PPO）需要模型本身具备一定的基础概率去命中正确答案，否则就会陷入“永远得不到正奖励”的死循环。STaR 通过“生成-过滤-反思”机制，用极低的成本为大模型注入了初始的 CoT（思维链）能力，为后续接入 GRPO 算法进行无止境的上限探索铺平了道路。

