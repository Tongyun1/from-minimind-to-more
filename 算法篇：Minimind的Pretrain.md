# 算法篇：Minimind的Pretrain

本文详细解析了 MiniMind 项目中的核心训练代码，涵盖数据处理、训练工具库以及预训练主程序。

要理解Minimind的训练方法，我们要重点关注三个文件：dataset/lm_dataset.py，trainer/trainer_utils.py，trainer/train_pretrain.py。

代码繁多，我们一步一步解析。

## 1. 数据管道层：`lm_dataset.py`

该文件定义了不同训练阶段所需的 `Dataset` 类，负责将原始文本转换为模型可读的 Tensor。

首先我们看看用于预训练的文本长什么样子。

```bash
(myminimind) root@GPU04:~/minimind$ head -n 2 /home/root/minimind/dataset/pretrain_hq.jsonl | jq
{
  "text": "鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> 好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？<|im_end|> 打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。<|im_end|> 为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。<|im_end|> 非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。<|im_end|> 帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。<|im_end|> 回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。<|im_end|> 识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。<|im_end|>"
}
{
  "text": "根据输入的内容，编写一个类别标签。\n这是一篇介绍如何阅读心电图的文章类别标签: 医学/心电图阅读指南<|im_end|> 帮我搜索一下最近的天气情况。当然，我可以帮您搜索最新的天气情况。请问您需要查询哪个城市的天气情况呢？<|im_end|> 帮我讲一个令人开心的笑话。好的，我帮您讲一个关于细菌的笑话。为什么细菌不会上网？因为连接总是断开了！<|im_end|> 现在给我生成一首关于大海的五言诗。碧波万顷月满天，海天相接处天地间。波涛滚滚江山美，海鸟翱翔日月闲。<|im_end|> 谢谢你，这篇文章很有用。不客气，我很高兴能够为您提供帮助。如果您还有其他问题或需求，随时可以对我说。<|im_end|> 你好，我想下载一个视频编辑软件，你有什么推荐吗？您好！当然，有很多选择。您想要免费软件还是愿意付费？<|im_end|> 为什么我的程序不输出正确结果？可能是代码逻辑有误，或者输入数据有误，需要仔细调试代码逻辑和输入数据。<|im_end|> 谢谢你的回答。现在我想知道这场比赛的具体时间和地点。这场比赛的时间是北京时间10月4日，地点是上海。<|im_end|>"
}
```

这份 `pretrain_hq.jsonl` 数据非常有趣。虽然它放在 Pretrain 阶段，但它的内容其实是 **QA 对话**。

这在业界被称为 **"Instruction Pre-training" (指令预训练)** 或者是用 Chat 数据进行 **Continued Pre-training**。

- **为什么这么做？**
  - 一般的 Pretrain 数据是维基百科、书籍（长篇大论）。
  - 这份数据是“一问一答”的短文本。
  - 把这些数据放到 Pretrain 阶段（不 Mask），是为了让模型在**大规模无监督学习**的时候，就提前熟悉 `<|im_end|>` 这种符号，熟悉“有人提问，紧接着就是回答”这种**文本概率分布**。

这将帮助你对模型训练中的数据流动建立初始的印象。接下来我们开始讲代码。

### 1.1 `PretrainDataset` (预训练数据集)

- **用途**：用于无监督预训练（Causal Language Modeling）。
- **逻辑**：
  1. 读取 JSONL 文件中的文本。
  2. 使用 Tokenizer 进行编码，并在首尾添加 `BOS` (Start) 和 `EOS` (End) 标记。
  3. **Padding**：将长度不足 `max_length` 的数据填充 `pad_token_id`。
  4. **Label 生成**：在预训练中，标签（Labels）与输入（Input IDs）完全一致（自回归任务），但在计算 Loss 时，模型会自动忽略 Pad 部分（通过设置 Padding 的 label 为 -100）。

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # [数据读取阶段] 使用 Hugging Face 的 datasets 库加载本地 JSONL 文件
        # 此时数据被加载为内存映射对象，self.samples[index] 会返回一个字典
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        # 返回整个数据集的样本总量
        return len(self.samples)

    def __getitem__(self, index):
        # 1. [获取原始文本] 根据索引从数据集提取出 'text' 字段的内容
        sample = self.samples[index]
        text_content = str(sample['text'])

        # 2. [分词与截断] 将文本转为数字 ID，预留 2 个位置给 BOS 和 EOS
        # truncation=True 保证长度不超过 max_length - 2
        tokens = self.tokenizer(
            text_content, 
            add_special_tokens=False, 
            max_length=self.max_length - 2, 
            truncation=True
        ).input_ids

        # 3. [添加特殊标记] 在文本前后分别加上“开始符”(BOS)和“结束符”(EOS)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 4. [填充 Padding] 如果长度不足 max_length，在后面补齐 pad_token_id
        # 最终得到一个长度固定为 max_length 的列表
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        # 5. [转为张量] 将 Python 列表转为 PyTorch 的长整型张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 6. [构造标签 Labels] 克隆一份 input_ids 作为训练的目标（预测下一个词）
        labels = input_ids.clone()

        # 7. [屏蔽 Padding] 将所有 Padding 部分的标签设为 -100
        # 在model_minimind.py中，使用PyTorch 的 CrossEntropyLoss 时，设置了忽略 -100 的位置，不计算 Loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 返回模型输入需要的 input_ids 和对应的训练标签 labels
        return input_ids, labels
```

我们可以在if \_\_name\_\_ == "\_\_main\_\_":中测试它：

```python
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/root/minimind/model")
    pretrainDataset = PretrainDataset(data_path='../dataset/sample_pretrain_hq.jsonl', tokenizer=tokenizer, max_length=500)
    print(len(pretrainDataset[0])) # 500
    print(pretrainDataset[0])
    
```

```bash
(myminimind) root@GPU04:~/minimind/dataset$ python lm_dataset.py 
/home/root/anaconda3/envs/myminimind/lib/python3.10/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
2
(tensor([   1,  168,  234,  115, 1415,  333, 1335,  413,  677, 2239,  269, 2397,
         315, 2669,  270, 1373, 4819,  496,  341, 1413,  965,  341,  677, 1486,
         508,  286,  590,  991, 1355, 1027, 2239, 3041, 3720,  168,  234,  115,
        1415, 2086, 2397,  315, 2669,  286,    2, 4520,  270, 2525, 1086,  397,
        1698, 2336, 4668,  269, 3858, 5027, 1355,   33, 4668,  269, 3858, 3897,
         724, 4222,  763, 2922,  286, 4600, 5156,  397, 1086,  608, 1698, 3445,
        2404,  371, 4222,  269, 3858, 3274,  814,    2,  223, 1970,  953,  457,
         120, 2941, 2121,  270,  627,  541,  909,  954, 4172,  568,  265,  228,
         829,  269,  457,  120, 2941,  286, 1188,  270,  397, 2434, 1086,  803,
        1970,  953,  457,  120, 2941, 2121,  270,  457,  120, 2941,  658,  368,
         909,  954, 4172,  568,  265,  228,  829, 1552,  507, 1211, 1122,  286,
           2,  223,  428, 1104, 3713, 1282,  333, 5565, 2208,  355,  541, 5962,
        2088,  269, 2604,  415, 4872,  368, 5874,  931, 1390,  230,  568, 1380,
         978, 2808, 1319,  286, 4374, 5962, 2088,  269, 2604,  415, 4872,  368,
        5874,  931, 1390,  230,  568,  416,  254, 1703, 2808,  496,  286,    2,
         223, 1101, 4176, 2155, 2212,  286, 1055, 4716,  270, 1142, 1327,  345,
        2345, 1541, 2722,  269,  814, 1142, 1327,  345, 2345, 1044, 4895, 1576,
         269, 3983, 2529,  415, 1413,  972, 1027, 6295,  269,  286,    2,  223,
        1086, 3589,  541, 4233, 3268,  232,  674,  286,  990,  550,  121, 4233,
         269,  355,    4, 1422, 2034, 5557, 2634,  269, 5839, 1002, 2465,    4,
         223, 1110, 3126, 3268,  232,  674, 2678, 2678,  549, 3504, 5044, 4710,
         104,  800,  619,  286,    2,  223, 2212,  541,  812,  270, 2261,  269,
        4249, 4101,  345, 4020,  814, 2261,  269, 4293, 4249, 4101, 2009,  428,
          24,   21,   25,   19,  770, 1267,  270, 3744, 2261,  544,  573,  100,
        1105,  536, 1346, 1554,  269, 3974,  269, 4293, 1258,  286,    2,  223,
        5159, 1966, 1446,  965,  963,  270,  581, 4180, 2467,  428, 1798,  714,
         102,  341,  714,  113, 3823,  341, 5958, 2922,  508,  286,  201, 1966,
        5355, 4668,  345, 3308,  492, 1395, 2207, 1025,  990, 1966,  269,  965,
         963,  345, 1798,  714,  102,  286,    2,    2,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([   1,  168,  234,  115, 1415,  333, 1335,  413,  677, 2239,  269, 2397,
         315, 2669,  270, 1373, 4819,  496,  341, 1413,  965,  341,  677, 1486,
         508,  286,  590,  991, 1355, 1027, 2239, 3041, 3720,  168,  234,  115,
        1415, 2086, 2397,  315, 2669,  286,    2, 4520,  270, 2525, 1086,  397,
        1698, 2336, 4668,  269, 3858, 5027, 1355,   33, 4668,  269, 3858, 3897,
         724, 4222,  763, 2922,  286, 4600, 5156,  397, 1086,  608, 1698, 3445,
        2404,  371, 4222,  269, 3858, 3274,  814,    2,  223, 1970,  953,  457,
         120, 2941, 2121,  270,  627,  541,  909,  954, 4172,  568,  265,  228,
         829,  269,  457,  120, 2941,  286, 1188,  270,  397, 2434, 1086,  803,
        1970,  953,  457,  120, 2941, 2121,  270,  457,  120, 2941,  658,  368,
         909,  954, 4172,  568,  265,  228,  829, 1552,  507, 1211, 1122,  286,
           2,  223,  428, 1104, 3713, 1282,  333, 5565, 2208,  355,  541, 5962,
        2088,  269, 2604,  415, 4872,  368, 5874,  931, 1390,  230,  568, 1380,
         978, 2808, 1319,  286, 4374, 5962, 2088,  269, 2604,  415, 4872,  368,
        5874,  931, 1390,  230,  568,  416,  254, 1703, 2808,  496,  286,    2,
         223, 1101, 4176, 2155, 2212,  286, 1055, 4716,  270, 1142, 1327,  345,
        2345, 1541, 2722,  269,  814, 1142, 1327,  345, 2345, 1044, 4895, 1576,
         269, 3983, 2529,  415, 1413,  972, 1027, 6295,  269,  286,    2,  223,
        1086, 3589,  541, 4233, 3268,  232,  674,  286,  990,  550,  121, 4233,
         269,  355,    4, 1422, 2034, 5557, 2634,  269, 5839, 1002, 2465,    4,
         223, 1110, 3126, 3268,  232,  674, 2678, 2678,  549, 3504, 5044, 4710,
         104,  800,  619,  286,    2,  223, 2212,  541,  812,  270, 2261,  269,
        4249, 4101,  345, 4020,  814, 2261,  269, 4293, 4249, 4101, 2009,  428,
          24,   21,   25,   19,  770, 1267,  270, 3744, 2261,  544,  573,  100,
        1105,  536, 1346, 1554,  269, 3974,  269, 4293, 1258,  286,    2,  223,
        5159, 1966, 1446,  965,  963,  270,  581, 4180, 2467,  428, 1798,  714,
         102,  341,  714,  113, 3823,  341, 5958, 2922,  508,  286,  201, 1966,
        5355, 4668,  345, 3308,  492, 1395, 2207, 1025,  990, 1966,  269,  965,
         963,  345, 1798,  714,  102,  286,    2,    2, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100]))
(myminimind) root@GPU04:~/minimind/dataset$ 
```

### 1.2 `SFTDataset` (有监督微调数据集)

- **用途**：用于指令微调（Instruction Tuning）。
- **核心难点**：**Loss Masking（损失掩码）**。在对话中，我们只希望模型学习“助手”的回答，而不学习“用户”的提问。
- **实现细节**：
  - `create_chat_prompt`：应用对话模板。
  - `generate_labels`：这是最关键的函数。它初始化全为 `-100`（忽略 Loss）的标签列表。然后扫描 Input IDs，找到“助手”回答的片段（介于 `bos_token`+`assistant` 和 `eos_token` 之间），将这部分的 Label 设为真实的 Token ID。
  - **效果**：模型只针对 Assistant 的输出计算梯度。

这是用于SFT的数据集处理，后面再讲。

### 1.3 `DPODataset` (偏好优化数据集)

- **用途**：用于 DPO (Direct Preference Optimization) 训练。
- **逻辑**：同时加载 `chosen`（人类偏好）和 `rejected`（人类拒绝）的数据对。返回成对的数据 `(x_chosen, y_chosen)` 和 `(x_rejected, y_rejected)` 及其对应的掩码。

这是用于DPO的数据处理，后面再讲。

## 2. 基础设施层：`trainer_utils.py`

该文件提供了训练所需的“脚手架”功能，特别是针对分布式训练和模型状态管理的封装。

### 2.1 `lm_checkpoint` (全能检查点管理)

这是一个非常健壮的保存/加载函数，解决了 PyTorch 原生保存的痛点：

- **DDP 解包**：在保存权重前，会自动检查模型是否被 `DistributedDataParallel` 包裹。如果是，则提取 `model.module` 或 `_orig_mod`，确保保存的是纯净的模型权重，方便后续推理或单卡加载。
- **通用性**：它不仅保存模型权重，还保存 `optimizer`（优化器状态）、`scaler`（混合精度缩放器）、`epoch`、`step` 和 `wandb_id`。这使得训练可以在中断后**完美恢复**。
- **区分保存**：
  - `checkpoint`: 包含所有训练状态，用于恢复训练。
  - `weight`: 仅包含模型参数，用于推理。

- **分布式自适应**：saved_ws != current_ws` 的逻辑非常硬核。它允许你灵活更换训练环境。比如你在 8 卡服务器上跑了一半，想拿回自己的双卡电脑接着跑，它能自动根据 GPU 总算力的变化帮你换算当前的训练步数。

- **原子化保存**：

  - 代码使用了 `os.replace(tmp, path)`。

  - **为什么要这样做？** 如果直接 `torch.save` 到目标文件，万一保存到一半断电了，你的旧 Checkpoint 就坏了，新 Checkpoint 也没存全。

  - **流程**：先写到一个临时文件 (`.tmp`) $\rightarrow$ 写完后再通过系统级指令瞬间改名覆盖。这保证了目标文件永远是完整的。

```python
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    # 确保保存目录存在，不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    
    # 根据是否使用 MoE (混合专家模型) 构造文件名后缀
    moe_path = '_moe' if lm_config.use_moe else ''
    # ckp_path: 仅保存模型权重（用于推理或分发）
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # resume_path: 保存完整训练状态（模型+优化器+进度，用于断点续训）
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    # --- [模式 A：保存 Checkpoint] ---
    if model is not None:
        # 1. 处理分布式训练和编译后的模型
        # 如果是 DDP 模型，取 .module；如果是 torch.compile 后的模型，取 ._orig_mod
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        
        # 2. 提取并优化 state_dict
        state_dict = raw_model.state_dict()
        # 关键步骤：转为半精度 (half) 并移至 CPU，节省磁盘空间且不占用显存
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        
        # 3. 安全保存模型权重 (使用 .tmp 中转防止保存过程中途崩溃导致文件损坏)
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        # 4. 获取 WandB 的 run_id，确保重启训练后日志能接上
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 5. 构建完整的恢复数据字典
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        # 6. 处理额外的需要保存的对象（如学习率调度器 scheduler）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'): # 如果对象有 state_dict，保存其状态
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 7. 安全保存恢复文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        
        # 8. 显存清理
        del state_dict, resume_data
        torch.cuda.empty_cache()
        
    # --- [模式 B：加载 Checkpoint] ---
    else: 
        if os.path.exists(resume_path):
            # 将数据加载到 CPU，避免撑爆主显卡显存
            ckp_data = torch.load(resume_path, map_location='cpu')
            
            # [核心逻辑] 处理 GPU 数量变化后的 Step 转换
            # 例如：之前用 2 张卡跑了 100 step，现在换成 4 张卡，
            # 为了保持数据消耗量一致，step 需要调整（100 * 2 // 4 = 50）
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None
```

### 2.2 `SkipBatchSampler` (断点续训采样器)

- **场景**：假设你在第 3 个 Epoch 的第 1000 个 Step 训练中断了。
- **作用**：当你重启训练时，`DataLoader` 通常会从头开始加载数据。这个自定义采样器会**跳过**前 1000 个 Batch 的数据索引，直接从第 1001 个 Batch 开始提供数据，确保模型不重复训练已见过的数据。

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        """
        初始化跳过批次的采样器
        :param sampler: 基础采样器（如 SequentialSampler 或 DistributedSampler），决定了索引的原始顺序
        :param batch_size: 批大小
        :param skip_batches: 需要跳过的 Batch 数量（通常从 checkpoint 中读取）
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []       # 用于暂存当前 Batch 的索引
        skipped = 0      # 记录已经跳过的 Batch 计数器
        
        # 遍历基础采样器产生的所有样本索引
        for idx in self.sampler:
            batch.append(idx)
            
            # 当收集的索引数量达到一个 Batch 大小时
            if len(batch) == self.batch_size:
                # 检查是否还需要继续跳过
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []   # 清空当前 Batch，但不返回给 DataLoader
                    continue
                
                # 如果已经跳够了，则产出该 Batch
                yield batch
                batch = []       # 重置 Batch 准备收集下一个
                
        # [处理尾部数据]
        # 如果样本总数不能被 batch_size 整除，处理最后剩下的不足一个 Batch 的样本
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        计算跳过之后，剩余的总 Batch 数量
        """
        # 1. 计算原始数据总共能分成多少个 Batch (向上取整)
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        # 2. 返回剩余的 Batch 数量，确保不为负数
        return max(0, total_batches - self.skip_batches)
```

### 2.3 `get_model_params` (参数统计)

- **MoE 支持**：精确计算参数量。它区分了：
  - **Total Params**：模型占用的总显存参数。
  - **Active Params**：推理时实际参与计算的参数（针对 MoE 模型，只统计被激活的专家参数）。

```pyhton
def get_model_params(model, config):
    # 1. [计算总参数量] 
    # 遍历模型所有参数张量，累加元素个数(numel)，除以 1e6 转换为“百万(M)”单位
    total = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. [获取 MoE 配置] 尝试从 config 读取混合专家模型的超参数
    # n_routed: 总共有多少个专家 (Routed Experts)，例如 64 个
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    # n_active: 每个 Token 实际激活选用的专家数 (Active Experts)，例如每次选 2 个
    n_active = getattr(config, 'num_experts_per_tok', 0)
    # n_shared: 共享专家数量 (Shared Experts)，这是 DeepSeek-MoE 等架构特有的，这些专家总是被激活
    n_shared = getattr(config, 'n_shared_experts', 0)

    # 3. [计算单个专家的参数量]
    # 技巧：通过筛选参数名中包含 'mlp.experts.0.' 的项，只统计“第0号专家”的大小
    # 假设所有专家的结构是一样的，算出一个就能代表所有
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    
    # 同理，计算单个“共享专家”的大小
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    # 4. [计算“骨架”参数 (Base Params)]
    # 骨架 = 总参数 - (单个路由专家 × 总数) - (单个共享专家 × 总数)
    # 这部分包含：Embedding, Attention, RMSNorm, OutputHead 等所有非 MLP 的公共部分
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # 5. [计算激活参数 (Active Params)]
    # 激活参数 = 骨架 + (单个路由专家 × 激活数) + (单个共享专家 × 总数)
    # 解释：推理时，Token 会经过骨架部分，经过所有共享专家，但在路由专家中只走 n_active 条路
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 6. [格式化输出]
    # 如果激活参数小于总参数，说明这是一个 MoE 模型
    if active < total: 
        # 输出格式例如：Model Params: 1000.00M-A200.00M (总参数10亿，实际激活2亿)
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: 
        # 如果是 Dense 模型（稠密模型），active == total
        Logger(f'Model Params: {total:.2f}M')
```

### 2.4 `get_lr` (余弦退火学习率)

- 实现了 Cosine Annealing 策略，学习率会随着 step 增加呈现余弦曲线下降，通常能带来比固定学习率更好的收敛效果。

```python
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```

## 3. 核心训练引擎：`train_pretrain.py`

这是整个训练流程的指挥中心，负责组装模型、数据和优化器，并执行训练循环。

### 第一阶段：环境初始化

1. **DDP 初始化**：`init_distributed_mode()` 检测是否多卡运行。
2. **随机种子**：`setup_seed()` 确保多卡训练的可复现性，且不同 GPU 使用不同的随机种子偏移。

```python
# ========== 1. 初始化环境和随机种子 ==========
# 初始化分布式环境（如果是多卡训练），获取当前进程的本地排名（local_rank，例如第0张卡还是第1张卡）
local_rank = init_distributed_mode()

# 如果分布式环境已初始化，指定当前进程使用的 GPU 设备
if dist.is_initialized(): 
    args.device = f"cuda:{local_rank}"

# 设置随机种子，确保可复现性。
# 关键点：每张卡使用不同的种子（42 + rank），防止所有卡生成完全一样的数据增强或 Dropout
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

### 第二阶段：模型与数据构建

1. **配置加载**：`MiniMindConfig` 初始化模型结构（是否使用 MoE，层数等）。
2. **断点检测**：
   - 如果 `args.from_resume=1`，尝试从 `checkpoints` 目录加载上次中断的完整状态（优化器、Step 等）。
   - 如果 `args.from_weight!='none'`，则仅加载预训练好的权重（用于微调）。
3. **WandB**：如果是主进程，初始化 WandB 用于可视化监控 Loss 和学习率。

```python
# ========== 2. 配置目录、模型参数、检查 Checkpoint ==========
    # 创建保存模型权重的文件夹
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 实例化模型配置类，决定模型大小（隐藏层大小、层数）和是否使用混合专家（MoE）
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # 检查是否有断点续训的需求（args.from_resume==1）
    # 如果有，尝试加载之前的 Checkpoint 数据（包含模型权重、优化器状态、step等）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度上下文 ==========
    # 判断使用 CPU 还是 GPU
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 选择半精度类型：优先使用 bfloat16（训练更稳定，范围更广），否则使用 float16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 创建自动混合精度（AMP）上下文管理器
    # 这将在后续的 forward 过程中自动将部分运算转为半精度以加速并节省显存
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 WandB 日志监控 ==========
    wandb = None
    # 仅在主进程（Master Node，通常是 rank 0）中初始化 WandB，避免多卡重复上传日志
    if args.use_wandb and is_main_process():
        import swanlab as wandb # 这里使用了 swanlab 作为 wandb 的替代或别名
        
        # 尝试从 Checkpoint 中恢复之前的 run_id，这样图表能接上
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        
        # 定义本次运行的名称，包含关键超参数
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        
        # 初始化项目
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型和分词器，如果指定了 args.from_weight，这里会加载预训练权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # [PyTorch 2.0 特性] 编译模型，优化计算图，显著提升训练速度
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # 加载数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 定义采样器：如果是分布式训练，必须使用 DistributedSampler 来确保每张卡分到不同的数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 定义梯度缩放器（Scaler），用于 float16 训练防止梯度下溢（bfloat16 通常不需要，但这里为了兼容写上了）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 定义优化器 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 Checkpoint 恢复完整训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 如果加载了 checkpoint，这里会将模型、优化器、Scaler 全部恢复到上次断掉的状态
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0) # 获取上次跑到了第几个 step
    
    # ========== 7. DDP 封装模型 ==========
    if dist.is_initialized():
        # 忽略旋转位置编码（RoPE）的预计算缓存，因为它们是常量，不需要同步梯度
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用 DDP 包装模型，负责多卡间的梯度同步
        model = DistributedDataParallel(model, device_ids=[local_rank])
```

### 第三阶段：核心训练循环 (`train_epoch`)

函数 `train_epoch` 包含具体的 Step 逻辑：

1. **动态学习率**：每个 Step 都调用 `get_lr` 更新学习率。
2. **前向传播 (Forward)**：
   - 使用 `autocast` 开启混合精度（bfloat16/float16），节省显存。
   - **Loss 计算**：`loss = res.loss + res.aux_loss`。这里显式加上了 MoE 的辅助 Loss（负载均衡 Loss），防止专家坍缩。
3. **反向传播 (Backward)**：
   - 使用 `scaler.scale(loss)` 放大 Loss，防止 FP16 下梯度下溢。
   - **梯度累积**：`loss / accumulation_steps`。这允许在小显存显卡上模拟大 Batch Size 训练。
4. **优化器步进**：
   - 每 `accumulation_steps` 步执行一次 `optimizer.step()`。
   - 执行 **梯度裁剪 (`clip_grad_norm_`)**，防止梯度爆炸。

```python
# ========== 8. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 关键步骤：每个 epoch 设置不同的随机种子，确保数据 shuffle 顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 再次设置 Python 随机种子（为了后续 indices 生成），并生成随机索引
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        
        # 计算需要跳过的步数：只有在“恢复训练的那个 epoch”才需要跳过之前的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # 使用自定义的 SkipBatchSampler，实现精确到 step 的断点续训
        # 它会快速空转跳过前 skip 个 batch，直接从断点处开始产出数据
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        # 定义 DataLoader
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        # 开始当前 Epoch 的训练
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 传入 start_step 是为了让进度条和日志显示的 step 数是正确的（累加的）
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
```

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行一个 Epoch 的训练流程。
    Args:
        epoch: 当前 Epoch 的索引。
        loader: 数据加载器 (DataLoader)。
        iters: 一个 Epoch 中包含的总 Iteration 数（步数）。
        start_step: 恢复训练时的起始步数。
        wandb: Weights & Biases 日志记录器对象。
    """
    start_time = time.time() # 记录 Epoch 开始时间，用于计算预计剩余时间 (ETA)
    
    # 遍历数据加载器
    # start=start_step + 1 确保显示的 step 计数是从断点后续开始的
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 1. 数据搬运：将数据移动到指定的计算设备（如 GPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 2. 学习率调度 (LR Scheduling)
        # 根据当前的总步数 (epoch * iters + step) 计算当前学习率
        # 这通常是 Cosine Annealing 或 Warmup 策略
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        
        # 手动更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 3. 前向传播 (Forward Pass) 与 混合精度 (AMP)
        with autocast_ctx: # 进入自动混合精度上下文（通常是 torch.cuda.amp.autocast）
            res = model(input_ids, labels=labels) # 模型前向推理
            
            # 计算总 Loss：主任务 Loss + 辅助 Loss (Auxiliary Loss)
            # Aux loss 常见于 MoE 模型（用于负载均衡）
            loss = res.loss + res.aux_loss
            
            # 梯度累积标准化：
            # 如果 accumulation_steps > 1，需要将 loss 除以步数，
            # 这样累积后的梯度和才是预期的均值，否则梯度会放大 accumulation_steps 倍
            loss = loss / args.accumulation_steps

        # 4. 反向传播 (Backward Pass)
        # scaler 用于处理 FP16 下的梯度下溢问题 (Gradient Underflow)
        scaler.scale(loss).backward()

        # 5. 梯度更新 (Optimizer Step) - 仅在达到累积步数时执行
        if (step + 1) % args.accumulation_steps == 0:
            # 先将梯度 unscale (反缩放) 回 FP32，以便进行梯度裁剪
            scaler.unscale_(optimizer)
            
            # 梯度裁剪 (Gradient Clipping)：防止梯度爆炸，稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新模型参数
            scaler.step(optimizer)
            
            # 更新 scaler 的缩放因子 (scale factor)
            scaler.update()

            # 清空梯度
            # set_to_none=True 比默认的 zero_grad() 更高效，因为它直接将梯度设为 None 而不是 0 张量
            optimizer.zero_grad(set_to_none=True)

        # 6. 日志记录 (Logging)
        # 每隔 log_interval 步 或 在最后一步时记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            
            # 还原真实的 Loss 数值用于显示 (乘以累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss # 纯预测任务的 Loss
            
            current_lr = optimizer.param_groups[-1]['lr']
            
            # 计算 ETA (预计剩余时间)
            # 公式: (已用时间 / 当前步数) * 总步数 / 60 - 已用时间 / 60
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印控制台日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            
            # 上传日志到 WandB
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 7. 模型保存 (Checkpointing)
        # 仅在主进程 (main process) 中保存，防止多卡训练时多个进程同时写文件导致冲突
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval() # 切换到评估模式 (影响 Dropout, BatchNorm 等)
            
            # 处理 MoE (Mixture of Experts) 特定的命名后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取原始模型对象，剥离 DDP 包装器 (DistributedDataParallel)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 处理 torch.compile 产生的 _orig_mod 包装
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            
            state_dict = raw_model.state_dict()
            
            # 保存模型权重：
            # 1. 转换为 half (FP16) 节省硬盘空间
            # 2. 移动到 CPU 防止占用显存
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 保存完整的训练状态 (包含 optimizer, scaler, epoch 等)，用于断点续训 (Resume)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            
            model.train() # 切回训练模式
            del state_dict # 显式删除字典，协助垃圾回收

        # 8. 激进的内存管理
        # 显式删除当前步的变量，防止引用计数导致显存无法释放
        # 这在显存紧张的 LLM 训练中非常常见
        del input_ids, labels, res, loss
```

### 第四阶段：清理

```python
# ========== 9. 清理分布式进程组 ==========
# 训练结束，销毁进程组，释放资源
if dist.is_initialized(): dist.destroy_process_group()
```

## 总结

Minimind的pretrain的代码大致就讲完了。然而，这份代码中藏着许多设计细节。请不要满足于跑通代码，而是逐行看懂，这样才能掌握它。在下一章中，我们将讲解Minimind的SFT算法设计。