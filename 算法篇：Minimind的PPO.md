# 算法篇：Minimind的PPO

## 引言

从这一章开始，我们真正进入了不使用监督学习方式，而以强化学习方式来提升模型能力的范畴（DPO实际上还是在监督学习标注数据）。为了让读者先对Minimind的PPO有一个大概的印象，我们通俗地将其概括为以下过程：

- 从SFT数据集中摘录了1万条高质量对话，构建约1MB大小的rlaif-mini.jsonl
- 模型对rlaif-mini.jsonl中的问题进行回答
- 使用InternLM2-1.8B-Reward对模型的回答进行打分
- 根据打分调整Minimind模型

我认为先讲理论后讲代码会让人晕头转向。在这里我们先：

- 一步一步把Minimind的PPO搞懂
- 讲讲这里的PPO与实际工业中的PPO有何异同
- 然后用理论去解释它为什么要这么做

## 数据准备：`lm_dataset.py`

首先还是老规矩，我们看看PPO用到的数据

```bash
(myminimind) root@GPU04:~/minimind$ head -n 2 /home/root/minimind/dataset/rlaif-mini.jsonl | jq .
{
  "conversations": [
    {
      "role": "user",
      "content": "列出五个基本的人格理论，并分别以一句话概括。"
    },
    {
      "role": "assistant",
      "content": "空"
    }
  ]
}
{
  "conversations": [
    {
      "role": "user",
      "content": "仔细阅读以下句子并回答“汤姆是医生还是建筑工人?”"
    },
    {
      "role": "assistant",
      "content": "空"
    }
  ]
}
```

可以看到它的格式与SFT的数据是相同的（本来就是从中摘录的），但是assistant并不需要内容，因为训练过程中完全由策略模型实时采样生成。RLAIF的训练过程中，模型会基于user的问题生成回答，然后由奖励函数/模型对回答打分， 分数高的回答会被鼓励（增加策略概率），分数低的回答会被抑制（降低策略概率）。这个"打分->调整"的循环就是强化学习的核心。

接下来我们看看数据集类的构建

```python
class RLAIFDataset(Dataset):
    # 初始化方法，在创建数据集实例时被调用
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        # 调用父类 (torch.utils.data.Dataset) 的初始化方法
        super().__init__()
        
        # 将传入的 tokenizer（分词器）保存为类属性，后续用于格式化和分词
        self.tokenizer = tokenizer
        
        # 保存序列的最大长度（注意：在这段代码的后续方法中，其实并没有真正用到这个属性）
        self.max_length = max_length
        
        # 使用 HuggingFace 的 datasets 库加载 JSONL 文件。
        # data_files 指定文件路径，split='train' 表示将其作为训练集加载
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        
        # 获取特殊 token 的 ID。这里尝试获取 "<BOS>assistant" 对应的 input_ids
        # add_special_tokens=False 表示不要自动在首尾添加额外的特殊符
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        
        # 获取结束符 (EOS token) 对应的 input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    # 返回数据集的总长度，这是 DataLoader 遍历数据时必需的
    def __len__(self):
        # 直接返回 samples（由 load_dataset 加载的数据集）的长度
        return len(self.samples)

    # 自定义辅助方法：将原始的对话列表转换为模型需要的 prompt 和目标 answer
    def create_chat_prompt(self, conversations):
        messages = [] # 用于存放标准格式的对话列表 (形如 [{"role": "user", "content": "..."}, ...])
        answer = ''   # 用于暂存最终的回复内容
        
        # 遍历原始对话列表，enumerate 会同时返回索引(i)和当前轮次的内容(turn)
        for i, turn in enumerate(conversations):
            # 根据索引的奇偶性判断角色。偶数轮(0, 2, 4...)是 user，奇数轮(1, 3, 5...)是 assistant
            # 这种写法强制假设对话是 user 和 assistant 严格交替进行的
            role = 'user' if i % 2 == 0 else 'assistant'
            
            # 按照 HuggingFace chat_template 要求的格式拼接字典，并加入 messages 列表
            messages.append({"role": role, "content": turn['content']})
            
            # 不断覆盖 answer，当循环结束时，answer 保存的就是对话列表中的最后一句话
            # （通常这一句应该是 assistant 给出的目标回复）
            answer = turn['content']
            
        # 使用 tokenizer 的 chat template 功能将历史对话格式化为单个字符串
        # messages[:-1] 切片表示：取除了最后一句话之外的所有对话（即历史上下文作为 Prompt）
        return self.tokenizer.apply_chat_template(
            messages[:-1], 
            tokenize=False,              # False 表示返回字符串，而不是 token IDs
            add_generation_prompt=True   # True 表示在字符串末尾自动加上让模型开始生成的引导符（如 "<|im_start|>assistant\n"）
        ), answer

    # 根据索引获取单条数据，这是 DataLoader 拉取数据的核心方法
    def __getitem__(self, index):
        # 取出指定索引处的原始数据字典
        sample = self.samples[index]
        
        # 提取出样本中的 'conversations' 字段，传给刚才写的辅助方法进行格式化
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        # 返回格式化后的结果。注意这里返回的是普通字符串字典，并没有将其转换为 Tensor
        return {
            'prompt': prompt,
            'answer': answer
        }
```

注意这里get_item返回的是普通字符串字典。我们打印出来看一下：

```python
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_root, "model"))

    rlaif_path = os.path.join(_root, "dataset", "rlaif-mini.jsonl")
    if not os.path.isfile(rlaif_path):
        print("未找到 rlaif-mini.jsonl:", rlaif_path)
        raise SystemExit(1)

    ds = RLAIFDataset(rlaif_path, tokenizer, max_length=1024)
    if len(ds) == 0:
        print("RLAIF 数据集为空")
        raise SystemExit(1)

    for i in range(min(2, len(ds))):
        print(ds[i])
```

```bash
(myminimind) zmm@GPU04:~/minimind/dataset$ python lm_dataset.py 
/home/zmm/anaconda3/envs/myminimind/lib/python3.10/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
{'prompt': '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n列出五个基本的人格理论，并分别以一句话概括。<|im_end|>\n<|im_start|>assistant\n', 'answer': '空'}
{'prompt': '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n仔细阅读以下句子并回答“汤姆是医生还是建筑工人?”<|im_end|>\n<|im_start|>assistant\n', 'answer': '空'}
```

这里没有做进一步的token化。本应在这里做的编码过程被移到了train_ppo.py中的ppo_train_epoch函数中。

## 训练代码主体`train_ppo.py`

PPO 训练中通常包含四个模型：Actor（当前策略模型）、Old Actor（旧策略模型，用于计算重要性采样比例）、Reference（参考模型，用于计算 KL 散度防止模型退化）和 Critic（价值模型，用于评估状态价值）。

这四个模型通常分为两组：**“正在训练的模型”** 和 **“冻结参数的模型”**。

**1. Actor Model**（演员模型 / 策略模型）

**状态： 可训练** (Trainable)

**来源：** 初始化自 SFT（Supervised Fine-Tuning）后的模型。

**作用：**这是我们最终想要得到的模型。它负责根据输入的 Prompt 生成回答（Action）。在训练过程中，它的参数会不断更新，目的是让生成的回答获得更高的奖励分数。



**2. Critic Model**（评论家模型 / 价值模型）

**状态： 可训练** (Trainable)

**来源：** 通常初始化自 Reward Model 或 SFT 模型（将最后的输出层改为标量回归头）。

**作用：**它是一个“价值函数”估计器（Value Function）。它不生成文本，而是接收当前的输入（Prompt + Actor 生成的部分回答），预估当前状态未来能获得多少总收益（Value）。

**核心功能：** 它的预测值用于计算**优势函数（Advantage）**，告诉 Actor 这一步走得是“好于预期”还是“差于预期”，从而指导 Actor 的梯度更新方向。



**3. Reward Model**（奖励模型）

**状态： 冻结** (Frozen)

**来源：** 在 RLHF 第二阶段，使用成对的人类偏好数据（Ranker）训练出来的模型。

**作用：**它是“考官”或“裁判”。当 Actor 生成一句完整的回答后，Reward Model 会给这个回答打一个分数（ScalarScore）。这个分数反映了回答符合人类偏好的程度。



**4. Reference Model**（参考模型）

**状态： 冻结** (Frozen)

**来源：** Actor Model 在开始 PPO 训练之前的完全拷贝（即 SFT 模型）。

**作用：**

它是“锚点”。**防止模型崩坏（Reward Hacking）**： 如果完全由 Reward Model 指挥，Actor 可能会为了高分输出乱码或钻空子。Reference Model 用来计算 **KL** **散度（**KL Divergence）。我们希望 Actor 在优化的同时，不要偏离原始 SFT 模型太远，保持语言的流畅性和基本能力。





![PPO框架图](./assets/PPO框架图.png)

以下是训练代码的详细解析。**请不要跳过，这里面有一些技巧是你需要看的。**

### 导入各种包

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel

# 导入自定义的模型配置、网络结构、数据集以及训练工具函数
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略所有的警告信息，保持控制台输出整洁
warnings.filterwarnings('ignore')
```

---

### Critic模型

```python
# 自定义的 Critic 模型（价值网络），继承自基础的语言模型 MiniMindLM
# 其作用是评估当前生成状态（State）的价值 V(s)
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 将原有的语言模型输出头（lm_head，输出词表大小）替换为一个线性层
        # 该线性层将隐藏状态映射为单一的标量值（即该状态的价值）
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 1. 前向传播：使用基础的 Transformer 模型获取所有 token 的隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 获取最后一层的隐藏状态并进行层归一化处理
        hidden_states = self.model.norm(outputs[0])
        # 2. 价值预测：将隐藏状态输入到 value_head 得到价值，并去掉最后一个维度 (B, SeqLen, 1) -> (B, SeqLen)
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

从这段 `CriticModel` 的代码本身来看，它输出的是**每个 token 的 value**（Token-level value），而不是直接输出一个代表整个回答的单一值。

1. `outputs[0]` 获取到的是整个序列的隐藏状态，它的维度是 `(Batch_size, SeqLen, Hidden_size)`。
2. 经过 `self.model.norm` 归一化后，维度保持不变。
3. 输入到 `self.value_head`（一个线性映射层）后，最后一个维度被映射为 1，此时维度变成了 `(Batch_size, SeqLen, 1)`。这意味着线性层对序列中的**每一个位置（token）**都独立计算出了一个标量价值。
4. 最后 `squeeze(-1)` 去掉了最后一个维度，返回值 `values` 的最终维度是 **`(Batch_size, SeqLen)`**。

虽然这个 Critic 模型对每一个 token 都算出了一个价值 $V(s_t)$，但在 `ppo_train_epoch` 训练函数中，作者在**实际使用**时只提取了代表“整个回答”的价值。

```python
# ppo_train_epoch节选
# 1. 这里得到了所有 token 的 value，形状为 [B, P+R] (即上面分析的 Batch_size, SeqLen)
values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  

# 2. 找到每个序列中“最后一个有效 token”的索引
last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)

# 3. 核心：只把最后一个 token 的 value 提取出来，形状变成了 [B]
values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  

# 4. 用这个提取出的值，与整个 response 的总 Reward 计算优势
advantages = rewards - values.detach()
```

**注意**，在标准的复杂 PPO 实现中（比如 HuggingFace 的 TRL 库），通常会利用完整的 `(B, SeqLen)` 输出去计算基于 Token 的广义优势估计 GAE (Generalized Advantage Estimation)。但 MiniMind 的代码为了极简，将整个生成过程视为一步（One-step MDP），所以只用了最后一个 token 的价值来对齐整体的 Reward。

---

### 标准PPO中的Critic模型

以下是标准 PPO 中 Critic 输出应用于 Actor 更新的完整工作流

1. 将奖励（Reward）分配到每个 Token

外部的 Reward Model 通常只给整段对话打一个最终分 $R_{final}$。在标准 PPO 中，这个分只给最后一个 token。同时，为了防止模型“胡言乱语”，会在**每一个 token** 处加上一个计算 Actor 和 Reference 模型差异的 KL 散度惩罚。

- 对于中间的 token ($t < T$)： 单步奖励 $r_t = -\beta \cdot \text{KL}_t$
- 对于最后一个 token ($t = T$)： 单步奖励 $r_T = R_{final} - \beta \cdot \text{KL}_T$

2. 获取 Critic 的每一个预测值 $V_t$

此时，你的 Critic 模型前向传播输出的 `values` 形状为 `[B, SeqLen]`。其中，第 $t$ 个位置的值 $V_t$ 代表：Critic 预测**从第 $t$ 个 token 开始直到句子结束，模型还能获得多少总奖励**。

3. 计算 Token 级别的优势（GAE）

这是标准 PPO 和你之前那份简化版代码最大的区别。标准做法使用 **广义优势估计（Generalized Advantage Estimation, GAE）**，为每一个 token 计算独立的优势 $A_t$。

**步骤 A：计算每个 token 的时序差分误差（TD Error）**

$$\delta_t = r_t + \gamma V_{t+1} - V_t$$

*(这里 $\gamma$ 是折扣因子，RLHF 中通常设为 1.0。如果当前 token 是最后一个，则 $V_{t+1} = 0$。)*

**步骤 B：逆序累加计算 GAE**

通过引入平滑参数 $\lambda$（通常为 0.95），将未来的 TD Error 衰减累加，得到每个 token 的最终优势：

$$A_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

4. Actor 计算 Token 级别的 Loss 并更新

现在，由于每个 token 都有了自己的优势 $A_t$，Actor 在计算 Loss 时不再是把整个句子打包，而是精细到每个词。

1. 计算每一个 token 动作的概率比率：

   $$ratio_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

2. 将比率与该 token 专属的 $A_t$ 相乘，并进行 PPO 的截断（Clip）：

   $$L_t = \min(ratio_t \cdot A_t, \text{clip}(ratio_t, 1-\epsilon, 1+\epsilon) \cdot A_t)$$

3. 最终 Actor 的 Loss 是对所有合法 token 的 $L_t$ 求平均，再进行反向传播。

---

### calculate_rewards函数

这是函数定义

```python
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
```

子函数，计算推理模型的奖励

```python
# 针对推理模型（Reasoning Model）的特殊奖励计算逻辑
    def reasoning_model_reward(rewards):
        # 1. 格式奖励：检查模型输出是否严格符合 <think>...</think><answer>...</answer> 的规范格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$" # 允许中间有空行

        # 使用正则匹配每个 response
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            # 如果符合标准格式，给予 0.5 的基础奖励，否则给 0.0
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        # 累加格式奖励到总奖励张量中
        rewards += torch.tensor(format_rewards, device=args.device)
```

子函数，计算标记奖励

```python
# 2. 标记奖励（Token Reward）：为了防止稀疏奖励导致难以训练，按出现特定的标签给予部分奖励
        def mark_num(text):
            reward = 0
            # 每出现一个正确的开始/结束标签，给予 0.25 奖励（最多集齐4个标签得到 1.0）
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        # 计算所有 response 的标记奖励并累加
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards
```

calculate_rewards函数

```python
# 初始化每个 response 的基础奖励为 0，维度为 [Batch_size]
    rewards = torch.zeros(len(responses), device=args.device)

    # 如果是训练推理模型，先加上基于规则的格式奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 3. 使用外部 Reward Model (奖励模型) 对语义和内容质量进行打分
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # 使用正则解析 prompt 中的对话历史 (system, user, assistant)
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 将当前模型生成的 response 作为 assistant 的回复拼接进去，构造完整的对话记录
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            # 调用奖励模型 API 计算得分。注意这里是对actor模型生成的整个句子进行打分。
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 限制得分在 [-3.0, 3.0] 范围内，防止异常大/小的 reward 导致训练崩溃
            scale = 3.0
            score = max(min(score, scale), -scale)

            # 如果是推理模型，不仅对整体打分，还要专门提取 <answer> 里的内容单独打分
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 针对纯答案部分单独计算 reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 加权合并：整体得分占 40%，最终答案得分占 60%
                    score = score * 0.4 + answer_score * 0.6
                    
            reward_model_scores.append(score)

        # 转换为张量并累加到总奖励上
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores # 形状为[B]

    return rewards
```

### ppo_train_epoch函数

请先阅读代码，若有疑问不要怕，先把代码看完，然后阅读下面的材料。

```python
def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """PPO 训练的一个 Epoch 主逻辑"""
    actor_model.train()
    critic_model.train()

    # 遍历 DataLoader 中的每个 batch
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # 获取当前批次的 prompt 列表 (List[str], 长度为 B)
        # 将文本 prompt 编码为 token，左侧填充（对于生成任务通常用左侧填充）
        # enc是BatchEncoding对象，包含input_ids和attention_mask。input_ids: [B, P], attention_mask: [B, P]. P是Prompt长度。这里的attention_mask为1的部分是Prompt部分，为0的部分是Padding部分。
        prompt_length = enc.input_ids.shape[1]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len, padding_side="left").to(args.device)
        prompt_length = enc.input_ids.shape[1]

        # ========== 1. Rollout（采样）阶段 ==========
        with torch.no_grad():
            # 如果使用 DDP，需通过 .module 才能调用 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            # Actor 模型根据 prompt 生成后续的 token
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # 生成结果形状：[B, Prompt_len + Response_len]

        # 将生成的 token 解码回纯文本 response (去除 prompt 部分)
        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        
        # ========== 2. 奖励与优势计算 ==========
        # 获取该 batch 生成结果的最终奖励分数 [B]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  

        # 生成 full_mask 以区分实际 token 和 padding token [B, P+R]。为1的位置是Prompt部分，为0的位置是Padding部分。
        full_mask = (gen_out != tokenizer.pad_token_id).long()  
        
        # Critic 模型计算整个序列每个位置的状态价值 [B, P+R]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  
        
        # 找到每个序列最后一个有效 token 的索引
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        # 提取最后一个有效状态的价值作为整个回复的预测价值 V(s) -> [B]
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  
        
        # 计算优势函数 (Advantage): A = Reward - Baseline(Value) -> [B]
        # .detach() 确保反向传播不会通过 values 影响此处的计算
        advantages = rewards - values.detach()  

        # ========== 3. PPO 策略网络（Actor）的前向传播 ==========
        with autocast_ctx: # 使用自动混合精度加速
            # 将生成的完整序列输入当前 Actor 模型，获取 logits
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, P+R, V]
            # 如果是 MoE 模型，获取辅助损失（用于负载均衡等），否则设为 0
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        
        labels = gen_out[:, 1:].clone()  # 偏移一位作为目标标签 [B, P+R-1]
        # 计算当前模型下，生成每个 token 的 log 概率：log P(a|s) -> [B, P+R-1]
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  
        
        # 构造掩码，仅对 "生成的 Response 部分" 的 logP 进行计算，忽略 Prompt 部分和 Padding 部分
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        
        # 当前策略下，整个 response 的总对数概率分布 [B]
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  

        # ========== 4. Old Actor 和 Ref Model 的前向传播 ==========
        with torch.no_grad():
            # 旧策略模型（用于 PPO 的重要性采样比率计算）
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # 参考模型（基座模型，用于防止策略过度偏移的 KL 惩罚计算）
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== 5. PPO 损失计算 ==========
        # KL散度： Actor 与 Old Actor 的分布差异 (监控指标)
        kl = (actor_logp - old_logp).mean()  
        # KL散度： Actor 与 参考模型 的分布差异 (直接作为惩罚项加入 loss)
        kl_ref = (actor_logp - ref_logp).mean()  
        
        # 计算重要性采样比率 (Importance Sampling Ratio): \pi_\theta / \pi_{old}
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        
        # PPO Clip 目标函数:
        surr1 = ratio * advantages  # [B] 未裁剪项
        # 裁剪项: 将 ratio 限制在 [1-epsilon, 1+epsilon] 之间
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        # Actor Loss: 负的最小化（即最大化目标函数）
        policy_loss = -torch.min(surr1, surr2).mean()  
        
        # Critic Loss: 预测的价值 Value 与实际 Reward 之间的均方误差 MSE
        value_loss = F.mse_loss(values, rewards)  
        
        # 总损失 = 策略损失 + 价值损失(乘以系数) + KL散度惩罚(乘以系数) + MoE辅助损失
        # 除以梯度累积步数以做平均
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps  
        
        # 反向传播计算梯度
        loss.backward()

        # ========== 6. 参数更新 ==========
        # 当达到梯度累积步数时，更新一次网络权重
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪，防止梯度爆炸
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            # 清空梯度
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ========== 7. 日志打印 ==========
        if is_main_process():
            # 计算生成的平均长度 (寻找 eos_token)
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            # 取出各项指标的值用于日志打印
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            # 上报 Wandb 平台
            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val, "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss, "reward": reward_val,
                    "kl": kl_val, "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val, "actor_lr": actor_lr,
                })

            # 控制台输出日志
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ========== 8. 模型状态同步 ==========
        # 定期用 Actor 的最新权重去覆盖更新 Old Actor 模型
        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            # 拷贝一份到 CPU 并加载至 old_actor，防止直接关联显存中的计算图
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # ========== 9. 模型保存 ==========
        # 定期保存，或者到达当前 Epoch 最后一个 iter 时保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 提取原模型（解除 DDP 包装）
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            # 转换成半精度存到硬盘，节省空间
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用工具函数保存完整的训练状态检查点（支持断点续训）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train() # 切回训练模式
            del actor_state # 手动释放内存

        # ========== 10. 内存清理 ==========
        # 删除不再需要的局部变量，防止每步训练后显存累积导致 OOM（内存溢出）
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss
```

---

**我知道直接看注释很难看懂，所以可以阅读以下材料。这是Gemini生成的，但我认为应该能解答你看代码过程中产生的疑问。**

#### 1. 准备工作：数据与编码

```python
prompts = batch["prompt"] 
enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                max_length=args.max_seq_len, padding_side="left").to(args.device)
```

- **在干什么**：把人类的文本提示（Prompt）转换成模型认识的数字 ID。
- **关键点**：注意 `padding_side="left"`。在做生成任务（文本续写）时，我们必须把填充用的 0 放在左边，保证所有 Prompt 的**最后一个有效 token 是对齐的在右侧**，这样模型才能正确地接着往下生成。

#### 2. Phase 1: Rollout（采样生成阶段）

```python
with torch.no_grad():
    gen_out = model_for_gen.generate(...)
responses_text = [tokenizer.decode(...) for i in range(len(prompts))]
```

- **在干什么**：这就是“做题”阶段。让当前的 Actor 模型根据 Prompt 自由生成回答。
- **关键点**：使用 `torch.no_grad()` 是因为这个时候我们只是为了**获取数据（经验）**，不需要计算梯度，这样能大大节省显存并加快速度。生成完毕后，把它解码回人类语言，准备送去打分。

#### 3. Phase 2: 奖励与优势计算 (Reward & Advantage)

```python
rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  
values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  
...
values = values_seq[..., last_indices]  
advantages = rewards - values.detach()  
```

- **在干什么**：
  1. 把生成的回答丢给外置的 `reward_model`，得到真实的分数（比如 2.5分）。
  2. 让 Critic 模型（价值网络）预测一下它觉得这个回答能得多少分（比如预测能得 1.5分）。
  3. 计算**优势 (Advantage)**：实际得分 - 预测得分 ($A = R - V$)。在这个例子中，优势是 1.0，说明模型表现得比预期好，是一个“正向惊喜”。
- **关键点**：`values.detach()` 非常重要。它切断了梯度的回传，因为 Critic 的价值预测仅仅是作为 Actor 计算优势的“基准线”，我们不希望 Actor 的更新去干扰 Critic 的参数。

#### 4. Phase 3: PPO 策略网络（Actor）的前向传播

```python
res = actor_model(input_ids=gen_out, attention_mask=full_mask)
logits = res.logits
...
actor_logp = (logp_tokens * final_mask).sum(dim=1)  
```

- **在干什么**：现在开始进入“反思”和“计算梯度”阶段。我们将刚才生成好的完整句子重新喂给 Actor 模型（这次去掉了 `no_grad`）。
- **目的**：计算模型生成刚才那个回答的**对数概率 (Log Probability)**。
- **关键点 - 掩码 (Mask)**：代码中花了很多笔墨构建 `final_mask`。这是因为整句话包含了 Prompt 和 Response。我们**绝对不能**因为 Response 生成得好，去更新 Prompt 部分的概率（因为 Prompt 是用户给的，不可控）。`final_mask` 确保了后续的更新**只针对模型自己生成的 Response 部分**。

#### 5. Phase 4: Old Actor 和 Ref Model 的前向传播

```python
with torch.no_grad():
    old_logits = old_actor_model(...)
    ref_logits = ref_model(...)
```

- **在干什么**：找两个“参照物”。
  - **Old Actor**：用来看看模型更新前，生成这段话的概率是多少。用于计算 PPO 的核心——重要性采样。
  - **Reference Model（初始基座模型）**：用来防止模型为了拿高分而“走火入魔”（比如无限重复某个得分高的词）。我们要求当前的策略不能偏离初始的聪明程度太远。

#### 6. Phase 5: PPO 损失计算 (核心数学逻辑)

```python
kl_ref = (actor_logp - ref_logp).mean()  
ratio = torch.exp(actor_logp - old_logp) 
surr1 = ratio * advantages  
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, ...) * advantages  
policy_loss = -torch.min(surr1, surr2).mean()  
value_loss = F.mse_loss(values, rewards) 
```

- **在干什么**：算总账，计算 Loss。
  - **Ratio**：当前概率 / 旧概率。如果比旧概率高，Ratio > 1。
  - **Surr1 / Surr2 (Clip 操作)**：这是 PPO 算法的灵魂。如果 Advantage 是正的，我们想提高这个动作的概率，但 `torch.clamp` 限制了 Ratio 最大只能是 $1 + \epsilon$（比如 1.1）。这就像是在说：“就算这次做得特别好，一次最多也只能进步 10%，不要步子迈太大扯着蛋（防止策略崩溃）。”
  - **Critic Loss**：就是让 Critic 预测的分数尽可能接近真实的 Reward（均方误差 MSE）。

#### 7. Phase 6: 参数更新

```
if (step + 1) % args.accumulation_steps == 0:
    clip_grad_norm_(...)
    actor_optimizer.step()
    ...
```

- **在干什么**：传统的深度学习更新套路。梯度裁剪（防爆炸） -> 优化器迈出一步修改权重 -> 调度器调整学习率 -> 清空梯度准备下一轮。

#### 8. Phase 7: 日志打印

- **在干什么**：算出当前生成句子的平均长度，把 Loss、Reward、KL散度等关键指标打印在屏幕上，并同步上传到 Wandb 平台。这是你监控训练有没有崩（比如 KL 突然爆炸、Reward 一直掉）的唯一途径。

#### 9. Phase 8 & 9: 状态同步与模型保存

- **在干什么**：
  - **同步**：每隔几个 step，就把当前不断进化的 Actor 的脑子（参数），复制一份给 Old Actor。这样 Old Actor 就能始终代表“前一刻的我”。
  - **保存**：定期把参数写进硬盘里（checkpoint），这样即使服务器突然断电，你也能从断点接着训。

#### 10. Phase 10: 内存清理 (非常关键)

```python
del enc, gen_out, responses_text, rewards ...
```

- **在干什么**：手动释放显存。
- **为什么**：RLHF 的一个 step 中既有推理（极度消耗显存），又有前向+反向传播。如果不立刻把这些巨大的张量（Tensor）从显卡里踢出去，两三个 batch 之后必然会触发臭名昭著的 `CUDA Out of Memory`。

---

#### 为什么采样的时候不算对数概率？

**1. 显存会瞬间爆炸 (OOM)** 

如果我们在 `generate` 阶段打开梯度去计算并记录每一步的对数概率，PyTorch 就必须在显存中保留整个自回归生成过程的**计算图（Computational Graph）**。 大模型生成 token 是一步一步来的（Autoregressive）。如果你生成了 1000 个 token，PyTorch 就要把这 1000 次前向传播的中间激活值（Activations）全部堆积在显卡里。对于几十亿参数的模型来说，哪怕 batch size 只有 1，显存也会瞬间撑爆。

**2. 批处理（Parallelism）效率极高** 

在 `generate` 阶段，我们只管生成文本，不计算梯度，这叫做“纯推理（Inference）”，速度最快、最省显存。 等句子全部生成完毕后，我们得到了完整的 `gen_out`（比如 1000 个 token）。此时，我们把这 1000 个 token 作为一个完整的序列，**一次性（并行地）喂给 Actor 模型。Transformer 架构的优势就在于并行计算，这样只需一次前向传播**就能得出所有 1000 个 token 的对数概率并构建好用于反向传播的计算图，效率远高于在生成时循环算 1000 次。

---

#### final_mask是如何构建的？

这是因果语言模型（Causal LM）训练中最容易绕晕的地方。因为预测下一个词存在一个**“错位（Shift）”**关系。

让我们用一个极简的例子，一步步推演代码里的矩阵运算。

假设：

- **Prompt**: `[A, B]` (长度 $P=2$)
- **模型生成的 Response**: `[C, D]`
- **为了对齐长度填充的 Padding**: `[<pad>]`
- 那么完整的序列 **`gen_out`**: `[A, B, C, D, <pad>]` (总长度 5)

我们的目标是：**只计算模型预测 `C` 和 `D` 时的 Loss。** 不算预测 `B` 的（那是 prompt），也不算预测 `<pad>` 的。

**第一步：错位生成标签 (Labels)**

```python
labels = gen_out[:, 1:].clone()  
# gen_out: [A, B, C, D, <pad>]
# labels:  [B, C, D, <pad>]  (长度变成了 4)
```

因为模型是用前一个词预测后一个词，所以对齐关系是：

- 输入 `A` $\rightarrow$ 预测 `B`
- 输入 `B` $\rightarrow$ 预测 `C` (这是 Response 的第一个词！)
- 输入 `C` $\rightarrow$ 预测 `D`
- 输入 `D` $\rightarrow$ 预测 `<pad>`

**第二步：构造 Prompt 掩码 (`resp_mask`)**

```python
seq_len = gen_out.size(1) - 1  # seq_len = 5 - 1 = 4
resp_mask = torch.arange(seq_len) >= prompt_length - 1
```

- `torch.arange(4)` 生成位置索引：`[0, 1, 2, 3]`

- `prompt_length` 是 2（即 `A, B`），所以 `prompt_length - 1 = 1`。

- 判断条件 `[0, 1, 2, 3] >= 1`，结果得到：

  **`resp_mask` = `[False, True, True, True]`**

我们把这个 Mask 和上面的预测关系对应起来看：

- 预测 `B` (索引0) $\rightarrow$ `False` (屏蔽！因为 B 是 Prompt)
- **预测 `C` (索引1) $\rightarrow$ `True` (保留！因为 C 是模型生成的起始词，它是由 Prompt 的最后一个词 B 预测出来的)**
- 预测 `D` (索引2) $\rightarrow$ `True` (保留！)
- 预测 `<pad>` (索引3) $\rightarrow$ `True` (保留！等一下，这里把 padding 也包进来了，所以需要下一步)

**这里最精妙的就是那个 `-1`**。正是因为“用当前词预测下一个词”的错位机制，Response 第一个词的预测其实发生在 Prompt 最后一个词的位置上。

**第三步：构造 Padding 掩码并合并 (`final_mask`)**

```python
# 找出标签中不是 pad 的位置
not_padding = ~labels.eq(tokenizer.pad_token_id)
# labels: [B, C, D, <pad>] -> not_padding: [True, True, True, False]

# 逻辑与运算：必须既属于 Response 范围，又不能是 Padding
final_mask = resp_mask & not_padding
```

进行最后一步合并：

- `resp_mask`:   `[False, True, True, True ]`
- `not_padding`: `[True,  True, True, False]`
- **`final_mask`**:  **`[False, True, True, False]`**

大功告成！这个 `final_mask` 完美地只剩下了中间两个 `True`，恰好对应我们要计算 Loss 的 `C` 和 `D` 两个 token。

通过这种矩阵运算构造 Mask，我们避免了在 Python 里写低效的 `for` 循环去找索引，极大地利用了 GPU 的并行张量计算能力。

#### 怎么又变成右填充了？

你肯定想起了我们在代码最开始看到的这一行：

```python
enc = tokenizer(prompts, ..., padding_side="left")
```

既然前面明确写了 **“左侧填充” (`padding_side="left"`)**，为什么在生成的完整序列 `gen_out` 里面，`<pad>` 却跑到了右边呢？

答案是：在一次完整的生成任务中，其实存在**两次不同方向的 Padding**。左边和右边的 Pad 承担着完全不同的使命。

我们把这个过程拆开来看：

**第一阶段：Prompt 的左侧填充（Left Padding）**

当我们在把 Prompt 喂给模型之前，因为一个 Batch 里不同的 Prompt 长度不一样，我们需要把它们对齐。 对于因果语言模型（Causal LM，比如 GPT 架构）来说，生成下一个词依赖于最后一个输入词。如果我们在右边填充，模型的注意力机制就会乱套，它会接着 `<pad>` 往下生成。 因此，**Prompt 必须左侧对齐**，保证所有有效 token 都挤在右边。

假设 Batch Size = 2：

- Prompt 1: `[A, B]` (长度 2)
- Prompt 2: `[X]` (长度 1) 经过 `padding_side="left"` 处理后，输入张量变成：
- Seq 1: `[ A, B ]`
- Seq 2: `[<pad>, X ]` 现在大家长度都是 2，模型可以同时开始从最右侧生成第一个词了。

**第二阶段：生成过程中的右侧填充（Right Padding）**

进入 `generate()` 函数后，模型开始一个词一个词地往外蹦。 由于不同句子的“话痨程度”不同，有的句子很长，有的句子很快就说完了（输出了 `<eos>` 结束符）。

假设模型要生成：

- Seq 1 生成了 3 个词：`[C, D, E]`
- Seq 2 生成了 1 个词就结束了：`[Y, <eos>]`

但是在 GPU 里，张量（Tensor）必须是一个完美的矩形矩阵。当 Seq 2 已经说完闭嘴了，而 Seq 1 还在继续生成 `D` 和 `E` 时，`generate()` 函数为了维持矩阵形状，就会**自动在 Seq 2 的右侧补上 `<pad>`**。

所以最终 `generate()` 吐出来的完整 `gen_out` 矩阵会是这样的：

- Seq 1: `[ A, B, C, D, E ]`
- Seq 2: `[<pad>, X, Y, <eos>, <pad>]`

回到问题本身

在我之前举的那个简化例子中： `gen_out: [A, B, C, D, <pad>]`

这里的 `<pad>` 正是**第二阶段生成完毕后，因为句子提前结束而填充在右侧的 Pad**。

我们在计算 Actor Loss 的时候：

1. **不想算 Prompt 的 Loss**：所以需要用 `resp_mask` 把前面的 `A`、`B`（包括左侧的 `<pad>`）给遮掉。
2. **不想算无意义填充的 Loss**：所以需要用 `not_padding` 把右边因为提前结束而多出来的 `<pad>` 给遮掉。

---

至此，最难的部分你已经看完了。下面是一些简单的主函数部分。

### 主函数

```python
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    # ... 省略部分参数声明注释，主要是超参配置
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=0, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式通信后端，获得当前设备的 local_rank
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，保证所有进程初始化一致
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 检查是否有之前保存的 checkpoint 用于断点续训
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 通过 Context Manager 设置自动混合精度（AMP），提升训练速度和降低显存消耗
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 Wandb (SwanLab) 监控平台 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化 PPO 的四个模型及外置奖励模型 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # 5.1 Actor模型（当前正在被训练的策略网络）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model) # 用 PyTorch 2.x 的 compile 加速计算图
        Logger('torch.compile enabled')
        
    # 5.2 Old Actor模型（提供旧策略的概率用于重要性采样，不计算梯度）
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    
    # 5.3 Reference模型（监督基线，用于计算 KL 散度，防止 Actor 退化，不计算梯度）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # 5.4 Critic模型（预测该状态能获得多少回报，独立更新权重）
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    # Critic 通常复用 Actor(即基座模型) 的隐层权重作为热启动
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    
    # 5.5 Reward模型（评估文本质量的外部模型，如 InternLM2-reward）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # ========== 6. 数据和优化器配置 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # Actor 和 Critic 分别使用独立的优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总迭代次数以初始化学习率调度器 (CosineAnnealingLR 余弦退火)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 7. 从 Checkpoint 恢复状态（如果启用了 from_resume） ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP 模型包装 ==========
    if dist.is_initialized():
        # 如果使用 RoPE (旋转位置编码)，需要忽略其缓存参数，防止 DDP 报错
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 分布式包装 Actor 和 Critic
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 9. 开始主训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist() # 打乱数据
        
        # 判断是否需要跳过已经训练过的 batch（针对断点续训）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        # 调用 PPO 训练逻辑
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 10. 训练结束，清理分布式进程组 ==========
    if dist.is_initialized(): dist.destroy_process_group()
```

### 归纳总结

下面，我再讲公式，应该就很明白了。

假设当前批次大小为 $B$，Prompt 长度为 $P$，生成的 Response 长度为 $R$。总序列长度为 $P+R$。

------

#### 第一阶段：Rollout（采样生成）

在这个阶段，模型只负责“闭卷考试”，不看答案，也不算分。

- **执行角色：** Actor（开启 `torch.no_grad()`）
- **输入：** Prompt 序列，维度 `[B, P]`
- **动作：** 自回归生成文本，直到遇到 `<eos>` 或达到最大设定的生成长度。
- **产出：** 完整的生成序列 `gen_out`。
- **数据粒度：** Token 级。序列维度变为 `[B, P+R]`。

#### 第二阶段：环境反馈（全局打分）

在这个阶段，“考官”对试卷进行总体评价。

- **执行角色：** Reward Model（无梯度，完全冻结）
- **输入：** 完整的自然语言文本（Prompt + Response）。
- **动作：** 根据人类偏好，给整个对话打分。
- **产出：** 全局标量奖励 $R_{final}$。
- **数据粒度：** 句子级（Sentence-level）。维度为 `[B]`。
- **注：** 在标准实现中，这个句子级的打分会被精准地放置在最后一个有效 Token（通常是 `<eos>`）的位置上，其余位置外部奖励均为 0。

#### 第三阶段：多模型并发前向（特征提取）

拿着完整的卷子，让各路模型给出自己对**每一个填空（Token）**的详细看法。输入全部是 `gen_out` `[B, P+R]`。

*(注意：经过 Mask 掩码处理后，以下所有 Token 级数据都会剔除 Prompt 和 Padding 部分，有效序列维度缩小为 `[B, R]`)*

- **Old Actor（旧策略快照） -> 无梯度**
  - 动作：评估“更新前的我”生成每个词的对数概率。这通常在 Rollout 刚结束时计算一次并缓存。
  - 产出：旧对数概率 $\log \pi_{old}(a_t|s_t)$。
- **Reference（基座模型） -> 无梯度**
  - 动作：评估“最原始的我”生成每个词的对数概率，作为防止模型严重退化的锚点。
  - 产出：参考对数概率 $\log \pi_{ref}(a_t|s_t)$。
- **Critic（价值模型） -> 有梯度，准备更新**
  - 动作：预测当前状态下，未来还能获得多少收益（Value）。
  - 产出：状态价值 $V_t$（严格记为 $V_\omega(s_t)$）。
- **Actor（当前策略） -> 有梯度，准备更新**
  - 动作：重新计算当前最新权重下，生成每个词的概率。在多次 PPO Epoch 循环中，这个值会不断变化。
  - 产出：当前对数概率 $\log \pi_{\theta}(a_t|s_t)$。

#### 第四阶段：核心指标计算 (GAE & 奖励塑形)

纯 CPU/GPU 上的张量数学计算，不涉及网络反向传播。

**1. KL 散度惩罚 (KL Penalty)**

通过动态或静态的惩罚系数 $\beta$，限制模型不要为了讨好 Reward Model 而输出乱码。

$$KL_t = \log \pi_{old}(a_t|s_t) - \log \pi_{ref}(a_t|s_t)$$

- **粒度：** Token 级，维度 `[B, R]`。

**2. 单步总奖励 (Step Reward $r_t$)**

将全局奖励 $R_{final}$ 融合进每一步的 KL 惩罚中。中间 Token 只有惩罚，只有最后一个 Token 能拿到全局奖励。

$$r_t = \begin{cases} - \beta KL_t & \text{if } t < T \\ R_{final} - \beta KL_T & \text{if } t = T \end{cases}$$

- **粒度：** Token 级，维度 `[B, R]`。

**3. 广义优势估计 (GAE, $A_t$)**

利用 Step Reward 和 Critic 的预测值，计算时序差分误差（TD Error $\delta_t$），然后从后往前逆序折现，评估每个 Token 的“相对优势”。其中 $\gamma$ 是折扣因子，$\lambda$ 是 GAE 平滑参数。

$$\delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)$$

$$A_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

- **粒度：** Token 级，维度 `[B, R]`。*(工程上还会对 $A_t$ 进行均值为0、方差为1的 Normalization 归一化)*

**4. 目标回报 (Target Returns $R_t$)**

用于指导 Critic 往哪更新的真实标签。

$$R_t = A_t + V_\omega(s_t)$$

- **粒度：** Token 级，维度 `[B, R]`。

#### 第五阶段：损失计算与反向传播 (Loss & Backprop)

将高维度的 Token 级数据坍缩为标量，指导神经网络的梯度下降。

**1. 策略损失 (Actor Loss)**

计算新旧策略的重要性比率（Ratio）。为了防止步子迈得太大导致训练崩溃，PPO 引入了截断参数 $\epsilon$（通常设为 0.2）。

$$ratio_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{old})$$

PPO-Clip 目标函数（注意，最小化 Loss，所以前面要加负号）：

$$L_{actor} = - \frac{1}{B \times R} \sum_{b=1}^{B} \sum_{t=1}^{R} \min \left( ratio_t(\theta) A_t, \text{clip}(ratio_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)$$

- **粒度压缩：** 从 `[B, R]` 的矩阵求平均（Mean），坍缩为标量 `[1]`，然后 `.backward()`。

**2. 价值损失 (Critic Loss)**

计算当前 Critic 预测值 $V_\omega(s_t)$ 与目标回报 $R_t$ 之间的均方误差 (MSE)。

$$L_{critic} = \frac{1}{B \times R} \sum_{b=1}^{B} \sum_{t=1}^{R} \frac{1}{2} (V_\omega(s_t) - R_t)^2$$

- **粒度压缩：** 同样从 `[B, R]` 求平均，坍缩为标量 `[1]`，然后 `.backward()`。

------

#### 💡 总结表

| **阶段 / 角色**      | **核心产出物理意义**        | **产出变量**            | **数据粒度 / Shape** |
| -------------------- | --------------------------- | ----------------------- | -------------------- |
| **Actor** (采样)     | 生成的完整文本              | `gen_out`               | Token 级 `[B, P+R]`  |
| **Reward Model**     | 句子的最终质量得分          | $R_{final}$             | 句子级 `[B]`         |
| **Old Actor** (缓存) | PPO重要性采样的分母         | $\log \pi_{old}$        | Token 级 `[B, R]`    |
| **Ref Model**        | 防止退化的基准线            | $\log \pi_{ref}$        | Token 级 `[B, R]`    |
| **Critic** (前向)    | 预测当前Token的未来收益     | $V_\omega(s_t)$         | Token 级 `[B, R]`    |
| **Actor** (前向)     | PPO重要性采样的分子         | $\log \pi_{\theta}$     | Token 级 `[B, R]`    |
| **数学计算** (GAE)   | 评估每个Token相对表现的优势 | $A_t$                   | Token 级 `[B, R]`    |
| **Loss** (降维)      | 网络更新的最终梯度方向      | $L_{actor}, L_{critic}$ | 标量 `[1]`           |

