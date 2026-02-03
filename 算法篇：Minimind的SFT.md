# 算法篇：Minimind的SFT

本文详细讲解了Minimind的SFT算法，包括数据构造，代码流程，训练工具库

还是从最关键的三个文件入手：trainer/train_full_sft.py，trainer/trainer_utils.py，dataset/lm_dataset.py。

我们这次先从核心训练文件开始。

## 1. 核心训练引擎：`train_full_sft.py`

事实上，SFT和pretratin的核心训练代码大同小异。主要原因是因为：

- 训练文件解决的主要问题是构造一个监督学习的框架，pretrain和SFT都是监督学习。
- 训练文件主要提供了各类训练工程问题的解决方法，而SFT 的核心逻辑（如何做 Masking，如何构建 Prompt）其实全在 Dataset 里。

下面是SFT训练代码以及详细注释，看过上一章的同学应该能比较轻易地看懂。

```python
import os
import sys

# 设置包名为 trainer，方便内部引用
__package__ = "trainer"
# 将上级目录加入 sys.path，确保能 import model 和 dataset 文件夹下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
# 导入自定义模块
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略一些不必要的警告（如 PyTorch 版本兼容性提示）
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    单个 Epoch 的训练逻辑
    :param epoch: 当前第几个 Epoch
    :param loader: 数据加载器
    :param iters: 总的 step 数量（用于计算剩余时间）
    :param start_step: 断点续训时的起始 step
    :param wandb: 在线日志记录对象
    """
    start_time = time.time()
    # enumerate 从 start_step+1 开始，配合 SkipBatchSampler 实现跳过已训练数据
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 1. 数据搬运到 GPU
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 2. 动态调整学习率 (Cosine Decay)
        # 根据当前总 step 计算 lr，Warmup 和 Decay 都在 get_lr 内部实现
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 3. 前向传播 (混合精度上下文)
        with autocast_ctx:
            # model 返回的是一个对象，包含 loss 和 aux_loss (MoE 负载均衡损失)
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # [面试考点] 梯度累加：Loss 必须除以累加步数，因为梯度是累加的，不除会导致梯度过大
            loss = loss / args.accumulation_steps

        # 4. 反向传播 (缩放 Loss 防止 FP16 下溢出)
        scaler.scale(loss).backward()

        # 5. 梯度更新 (达到累加步数才更新)
        if (step + 1) % args.accumulation_steps == 0:
            # 先将梯度 unscale 回原来的大小
            scaler.unscale_(optimizer)
            # [面试考点] 梯度裁剪：防止梯度爆炸 (Exploding Gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            scaler.update() # 更新 scaler 的缩放因子

            # 清空梯度 (set_to_none=True 比默认的 False 更快，减少内存操作)
            optimizer.zero_grad(set_to_none=True)

        # 6. 日志打印 (Log)
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原真实的 Loss 值用于显示 (乘以 accumulation_steps)
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss # 纯语言模型 Loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算预计剩余时间 (ETA)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 7. 模型保存 (Checkpoints)
        # 仅主进程 (is_main_process) 负责保存，避免多进程同时写入文件冲突
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval() # 切换到评估模式 (关闭 Dropout 等)
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 处理 DDP 模型包裹：如果是 DDP，需要取 .module 拿到原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 处理 torch.compile 包裹：如果是编译后的模型，取 _orig_mod
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            
            state_dict = raw_model.state_dict()
            # 保存权重为半精度 (half) 以节省空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 保存包含 optimizer 状态的完整 checkpoint，用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train() # 切回训练模式
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    # ========== 0. 参数解析 (Argparse) ==========
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型权重(仅权重)保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存文件的前缀")
    parser.add_argument("--epochs", type=int, default=2, help="训练总轮数，SFT通常轮数较少")
    parser.add_argument("--batch_size", type=int, default=16, help="单卡 Batch Size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="SFT 学习率通常比 Pretrain 小")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="指定主设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="BF16 训练更稳定，防止溢出")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 进程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累加步数，用于显存不足时模拟大 Batch")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="每隔多少步打印日志")
    parser.add_argument("--save_interval", type=int, default=1000, help="每隔多少步保存模型")
    parser.add_argument('--hidden_size', default=512, type=int, help="模型维度，需与 Pretrain 一致")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--max_seq_len', default=340, type=int, help="SFT 数据截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否启用混合专家模型")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl", help="SFT 数据集路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="加载的预训练权重文件名")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从 Checkpoint 恢复训练")
    parser.add_argument("--use_wandb", action="store_true", help="启用 WandB 监控")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="WandB 项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="使用 PyTorch 2.0 编译加速")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境与随机种子 ==========
    # DDP 初始化：设定 RANK, WORLD_SIZE 等环境变量
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 固定种子，保证多卡训练时模型初始化一致，且 Shuffle 逻辑可复现
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置与环境准备 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 实例化模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果是断点续训 (from_resume=1)，则加载 Checkpoint 元数据
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 (AMP) ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # bfloat16 范围更广，float16 精度更高但易溢出。A100/H100 推荐 bfloat16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 WandB (SwanLab) ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb # 这里使用了 swanlab 作为 wandb 的替代品
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型，并加载预训练权重 (from_weight)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model) # 图编译加速
        Logger('torch.compile enabled')
    
    # SFT 数据集处理 (Masking 逻辑在 SFTDataset 内部)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # GradScaler 用于 AMP 训练中放大 Loss，防止梯度下溢变为 0
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 断点续训逻辑 (Resume) ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重、优化器状态、Scaler 状态
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    # ========== 7. DDP 模型封装 ==========
    if dist.is_initialized():
        # 忽略不需要同步梯度的 buffer (如 RoPE 的 sin/cos 表)
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 需要 set_epoch 保证每个 epoch 的 shuffle 随机数不同
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 重新生成索引，用于 SkipBatchSampler
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是当前断点的 epoch，则计算需要跳过多少个 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # SkipBatchSampler：自定义 Sampler，直接跳过前 N 个数据，避免 DataLoader 空转
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理资源 ==========
    if dist.is_initialized(): dist.destroy_process_group()
```

## 2. 数据管道层:`lm_dataset.py`

接下来讲SFT的重点。

首先，我们还是查看一下sft数据的构造，这能让你对sft在做什么构建基本印象。

```bash
(myminimind) root@GPU04:~/minimind$ head -n 2 /home/root/minimind/dataset/sft_512.jsonl | jq
{
  "conversations": [
    {
      "content": "给定一段文本和关键词列表，删除文本中包含所有给定关键词的子字符串。\n文本：\"这是一个测试句子，目的是看看模型是否可以正确地从这个句子中删除关键词。\"\\n关键词列表：[‘测试’，‘模型’]",
      "role": "user"
    },
    {
      "content": "删除包含所有给定关键词的子字符串后，文本变为：\"这是一个句子，目的是看看是否可以正确地从这个句子中删除关键词。\"",
      "role": "assistant"
    },
    {
      "content": "好的。现在请你将这个文本中的所有的逗号都替换成空格。",
      "role": "user"
    },
    {
      "content": "好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为：\"这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。\"。处理结果如何？",
      "role": "assistant"
    }
  ]
}
{
  "conversations": [
    {
      "content": "写一个关于秋天的诗吧。",
      "role": "user"
    },
    {
      "content": "红黄绿交织，田野金黄色。风吹枫叶落，人在秋天慢步行。",
      "role": "assistant"
    },
    {
      "content": "给我推荐一些好听的摇滚歌曲。",
      "role": "user"
    },
    {
      "content": "摇滚歌曲推荐：AC/DC的Back in Black，Guns N' Roses的Sweet Child O' Mine，Rolling Stones的Satisfaction，Led Zeppelin的Stairway to Heaven，Nirvana的Smells Like Teen Spirit等。",
      "role": "assistant"
    },
    {
      "content": "计算一下2的100次方。",
      "role": "user"
    },
    {
      "content": "2的100次方等于1.267650600228229401496703205376e+30，约等于1.27乘以10的30次方。",
      "role": "assistant"
    },
    {
      "content": "帮我回答一道历史题目。清朝时期的八旗共有多少旗人？",
      "role": "user"
    },
    {
      "content": "清朝时期八旗旗人总数约为200万人左右，其中正黄旗、正蓝旗、正白旗、正红旗的人数较多，其他旗的人数较少。",
      "role": "assistant"
    }
  ]
}
```

我们可以看到SFT的数据和pretrain的数据有比较大的差别。

- **结构**：`{"conversations": [...]}` 是一个列表。
- **逻辑**：列表中的元素的顺序代表了**时间顺序**。
- **Sample 1 （连贯上下文）**：
  - 用户先做了一个“删除关键词”的任务。
  - **关键点**：第三轮 `User: "好的。现在请你将这个文本..."` 这是一个**依赖上下文**的指令。“这个文本”指代的是上一轮 Assistant 输出的结果。
  - **SFT的作用**：让模型学会维护 Context（上下文窗口），知道“这个”指代的是什么。
- **Sample 2 （综合能力）**：
  - 包含写诗、推荐歌曲、数学计算、历史问答。
  - 这是一个典型的**通用能力（General Capability）**训练样本，目的是让模型在同一个 Session 中能应对话题的快速切换。

这是面试中的**核心考点**。虽然 JSON 里 `user` 和 `assistant` 是分开的，但在送入模型训练时，它们会被“压扁”成**一条长序列**。

lm_dataset.py:

```python
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_root, "model"))

    sft_path = os.path.join(_root, "dataset", "sample_sft_512.jsonl")
    if not os.path.isfile(sft_path):
        print("未找到 sft_512.jsonl:", sft_path)
        raise SystemExit(1)

    ds = SFTDataset(sft_path, tokenizer, max_length=1024)
    if len(ds) == 0:
        print("SFT 数据集为空")
        raise SystemExit(1)

    for i in range(min(2, len(ds))):
        sample = ds.samples[i]
        prompt = ds.create_chat_prompt(sample["conversations"])
        print("--- Sample {} ---\n{}\n".format(i, prompt))
```

可以看到，在对sft的数据应用create_chat_prompt()之后，数据被“压扁”成了类似pretrain数据的样子

```bash
(myminimind) root@GPU04:~/minimind/dataset$ python lm_dataset.py 
/home/root/anaconda3/envs/myminimind/lib/python3.10/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Generating train split: 100 examples [00:00, 23255.18 examples/s]
--- Sample 0 ---
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
给定一段文本和关键词列表，删除文本中包含所有给定关键词的子字符串。
文本："这是一个测试句子，目的是看看模型是否可以正确地从这个句子中删除关键词。"\n关键词列表：[‘测试’，‘模型’]<|im_end|>
<|im_start|>assistant
删除包含所有给定关键词的子字符串后，文本变为："这是一个句子，目的是看看是否可以正确地从这个句子中删除关键词。"<|im_end|>
<|im_start|>user
好的。现在请你将这个文本中的所有的逗号都替换成空格。<|im_end|>
<|im_start|>assistant
好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为："这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。"。处理结果如何？<|im_end|>


--- Sample 1 ---
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
写一个关于秋天的诗吧。<|im_end|>
<|im_start|>assistant
红黄绿交织，田野金黄色。风吹枫叶落，人在秋天慢步行。<|im_end|>
<|im_start|>user
给我推荐一些好听的摇滚歌曲。<|im_end|>
<|im_start|>assistant
摇滚歌曲推荐：AC/DC的Back in Black，Guns N' Roses的Sweet Child O' Mine，Rolling Stones的Satisfaction，Led Zeppelin的Stairway to Heaven，Nirvana的Smells Like Teen Spirit等。<|im_end|>
<|im_start|>user
计算一下2的100次方。<|im_end|>
<|im_start|>assistant
2的100次方等于1.267650600228229401496703205376e+30，约等于1.27乘以10的30次方。<|im_end|>
<|im_start|>user
帮我回答一道历史题目。清朝时期的八旗共有多少旗人？<|im_end|>
<|im_start|>assistant
清朝时期八旗旗人总数约为200万人左右，其中正黄旗、正蓝旗、正白旗、正红旗的人数较多，其他旗的人数较少。<|im_end|>

```

**注意，这里有SFT 和 Pre-train 最大的区别。**

- **Pre-train**：计算所有 token 的 Loss。
- **SFT**：**只计算 Assistant 回复部分的 Loss**。

对应到数据：

1. **User 说的话** ("写一个关于秋天的诗吧。") -> **Label设为 -100** (不计算梯度)。
2. **Assistant 说的话** ("红黄绿交织...") -> **Label设为 Token ID** (计算梯度)。
3. **User 说的话** ("给我推荐一些...") -> **Label设为 -100**。
4. **Assistant 说的话** ("摇滚歌曲推荐...") -> **Label设为 Token ID**。

下面是SFTDataset的详细注释

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化数据集
        :param jsonl_path: SFT 数据文件路径 (jsonl 格式)
        :param tokenizer: 分词器
        :param max_length: 单条数据的最大长度 (超过会被截断)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # [核心考点 1: 懒加载]
        # 使用 HuggingFace 的 load_dataset 加载 JSONL。
        # 它的优势是 Lazy Loading（懒加载）和 Memory Mapping（内存映射）。
        # 即便 jsonl 文件有几 GB，也不会一次性读入内存，而是用到哪条读哪条。
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        
        # [核心考点 2: 定位特征码]
        # 为了实现“只对 Assistant 的回复计算 Loss”，我们需要在长文本中找到回复的起始和结束位置。
        # 这里预先计算好“Assistant起始符”和“结束符”对应的 Token ID 序列。
        # 注意：add_special_tokens=False 很重要，因为我们不想要 BOS/EOS 再次包裹这些片段。
        
        # 假设模板是 ChatML，这里硬编码了 'assistant\n'。
        # 风险提示：如果你的 Chat Template 渲染出来是 '<|im_start|>assistant' (没换行)，
        # 这里的匹配逻辑就会失效，导致 Loss 全为 0。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        # 返回总样本数
        return len(self.samples)

    def create_chat_prompt(self, cs):
        """
        [核心考点 3: 格式化]
        将多轮对话列表转换为纯文本字符串。
        例如：[{"role": "user", "content": "hi"}] -> "<|im_start|>user\nhi<|im_end|>\n..."
        """
        messages = cs.copy()
        # 简单的工具调用兼容逻辑（如果有 function call）
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        
        # 使用 Tokenizer 自带的 apply_chat_template 进行渲染
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # 关键：这里只拼成字符串，先不转 ID，方便后续统一截断和处理
            add_generation_prompt=False, # SFT 是训练已有对话，不需要像推理时那样自动添加 "assistant:" 引导头
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        [核心考点 4: Loss Masking / Label Masking]
        这是 SFT 代码的灵魂。生成与 input_ids 等长的 labels 序列。
        规则：
        - User 的话 -> 设为 -100 (PyTorch CrossEntropyLoss 默认忽略 -100)
        - Assistant 的话 -> 设为原本的 Token ID (参与计算梯度)
        - Padding -> 设为 -100
        """
        # 1. 初始化全为 -100（默认全不学）
        labels = [-100] * len(input_ids)
        i = 0
        
        # 2. 线性扫描 input_ids，寻找 Assistant 的回复区间
        while i < len(input_ids):
            # 判断当前位置是否匹配“Assistant起始符” (self.bos_id)
            # 例如：检查 input_ids[i] 是否等于 <|im_start|> 且 input_ids[i+1] 是否等于 assistant...
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 找到了 Assistant 说话的开头
                start = i + len(self.bos_id)
                end = start
                
                # 继续向后找，直到找到“结束符” (self.eos_id)
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 3. [解除 Mask]
                # 将 start 到 end 之间的部分（即回复内容）从 -100 恢复为真实的 input_ids
                # 只有这一部分会产生梯度，更新模型参数
                # min(..., self.max_length) 是防止越界
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                
                # 移动指针 i 到当前回复结束的位置，继续找下一轮（多轮对话场景）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 如果没匹配到，指针后移一位
                i += 1
        return labels

    def __getitem__(self, index):
        """
        DataLoader 获取单条数据的入口
        """
        # 1. 获取原始数据
        sample = self.samples[index]
        
        # 2. 渲染 Prompt 字符串
        prompt = self.create_chat_prompt(sample['conversations'])
        
        # 3. 分词 (Tokenization) 并截断
        # 只取前 max_length 个 token，超出的直接丢弃
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # 4. [核心考点 5: Padding]
        # 如果长度不够 max_length，用 pad_token_id 补齐
        # 这样做是为了让一个 Batch 内的数据维度一致，才能堆叠成 Tensor
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 5. 生成 Mask 后的标签
        labels = self.generate_labels(input_ids)
        
        # # === 调试代码 (强烈建议在正式训练前取消注释跑一次) ===
        # # 作用：人工肉眼检查 Mask 是否正确。
        # # 也就是看 User 的部分对应的 Label 是否真的是 -100。
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     # 打印 Input Token 和 对应的 Label (注意这里模拟了 Next Token Prediction 的错位)
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================================================
        
        # 6. 转为 Tensor 返回
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

## 3. 基础设施层：`trainer_utils.py`

已经在《算法篇：Minimind的Pretrain》讲过了，这里不在赘述。

## 4. 总结

至此，我们已经完整学完了 MiniMind 从 **Pre-train（预训练）** 和 **SFT（指令微调）**。

**Pre-train **让模型在海量的文本海洋中冲浪，通过“根据前文预测下一个词”的自监督任务，学习人类语言的统计分布、语法逻辑以及蕴含在文字背后的世界知识。这是模型世界知识的来源，但此时它还只是一个只会续写的“复读机”，不懂得如何听从指令。

**SFT **通过人工标注的对话数据，教导模型理解“问题”与“回答”的对应关系。通过对 User 部分进行 Loss Masking，我们强迫模型只针对 Assistant 的回复产生梯度更新。这让模型从一个原始的语言概率模型，变为一个能和人沟通的助手。

**下一阶段，我们将学习 MiniMind 实现的各类 RL（强化学习）算法**。如果说 SFT 是通过“模仿”标准答案来学习，那么 RL 则是让模型在“尝试与反馈”中进化。RL 算法是近期大模型领域的绝对核心，无论是传统的 **PPO**，还是目前大火的 **DPO (Direct Preference Optimization)** 甚至是针对推理能力强化的 **GRPO**，其本质都是为了解决“标准答案无法覆盖所有场景”的问题。

RL 算法通过引入奖励模型（Reward Model）或偏好对齐，进一步提升模型的：

- **对齐性（Alignment）**：确保模型的价值观与人类逻辑一致，不产生幻觉或有害内容。
- **推理上限（Reasoning Cap）**：通过思维链（CoT）与自我博弈（Self-play），让模型在数学、逻辑等高难度任务上突破 SFT 的模仿天花板。
- **鲁棒性（Robustness）**：使模型在面对未见过的刁钻提问时，依然能给出稳定、高质量的回答。

掌握了 RL，才算真正触碰到了当前大模型最前沿的“灵魂注入”工程。

## 作者的思考

我们知道人类在幼年时期会经历一个强化学习过程，以习得各项技能，而在幼年时期没有得到良好训练的人会表现出各种问题。比如：如果不教一个幼儿说话，那么他未来很有可能面临严重的语言障碍；亦或者阻止幼儿自发性的“摔东西”的行为，那么他习得空间感知能力的过程就会延长。迁移到LLM领域，很多人产生一种观点，即RL可以有效地提升LLM的能力，甚至有可能成为AGI的关键拼图。

然而最近《Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?》（NeurIPS 2025 Best Paper Runner-Up Award，ICML 2025 Workshop AI4Math Best Paper Award）这篇文章通过大量的实验验证了一个观点，即：RL只是提高了LLM的采样效率，而不是真正让模型学会了不会做的事。

那么，到底要怎样才能让模型像人类幼儿一样真正地学习呢？我认为目前我们对RL的认识还极为有限，这个领域还有很多未探索的宝藏等待我们发掘。





































