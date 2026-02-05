# 算法篇：Minimind的DPO

由于DPO本质上还是在做一个交叉熵损失，所以代码与SFT是比较相似的。

主要需要关注的代码有：trainer/train_dpo.py，dataset/lm_dataset.py，trainer/trainer_utils.py

## 1. 数据管道层:`lm_dataset.py`

这次我们还是先从数据开始看。处理数据的方式包含了DPO的核心逻辑。

```python
(base) root@GPU04:~/minimind$ head -n 1 /home/root/minimind/dataset/dpo.jsonl | jq
{
  "chosen": [
    {
      "content": "continue",
      "role": "user"
    },
    {
      "content": "As Recharge Retreats grows, we plan to expand our team with additional event coordinators and marketing specialists. To accommodate this growth, we will establish an office space that fosters collaboration and creativity among team members. We will also prioritize remote work options and streamlined communication tools to support remote team members.\n\nIn addition, we aim to diversify our retreat offerings to cater to different themes, such as solo travel retreats, wellness retreats for parents, and spiritual journeys. We will seek to partner with multiple venues across various locations worldwide to provide a diverse range of retreat experiences.\n\nWe will also expand our marketing efforts to reach a wider audience, maximizing brand exposure through influencer partnerships, experiential marketing, and virtual events. We will continuously strive to refine our services based on client feedback and identify industry trends to adapt our offerings to the ever-changing needs and preferences of our target market.\n\nAs we scale, we will remain committed to upholding our mission to empower and connect women through transformative retreat experiences that prioritize personal growth, wellness, and connection.",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "continue",
      "role": "user"
    },
    {
      "content": "A. Scaling:\n\n1. Offer a franchise model: Allow others to open and operate Recharge Retreats franchises in different regions, expanding our reach and revenue potential.\n2. Develop online courses and coaching programs: Create supplementary income streams by offering training and support to individuals and organizations interested in organizing their own retreats.\n\nB. Diversification:\n\n1. Expand service offerings: Consider adding additional retreat-related services, such as coaching, ongoing support, retreat evaluation, or retreat financing options.\n2. Target new markets: Broadening our target market to include other demographics, such as men, families, or specific interest groups.\n\nC. International Expansion:\n\n1. Partner with local organizations: Collaborate with women's organizations abroad to plan international retreats and expand our global presence.\n2. Seek funding opportunities: Pursue grants, investments, and loans to finance international retreats and support expansion in new markets.\n\nBy following this business plan, Recharge Retreats has the potential to become a leading provider of women's retreat services, empowering individuals and organizations to foster growth, connection, and personal transformation through memorable and life-enhancing experiences.",
      "role": "assistant"
    }
  ]
}
```

可以看到DPO的每一条数据由偏序对组成：一个chosen和一个rejected分别代表我们认为好的回答和坏的回答。

这种数据格式的核心目的是告诉模型：**“在面对同样的输入（Prompt）时，回答 A 比回答 B 更好。”**

JSON 对象包含两个核心键（Key）：

- `"chosen"` (胜者/正样本)：代表人类标注员或强模型认为更好的回答。
- `"rejected"` (败者/负样本)：代表被认为较差的回答。

两者的内部结构完全一致，都是一个**多轮对话列表**（List of Messages）：

- `role: "user"`：用户输入。在这里，chosen 和 rejected 的用户输入**必须完全一致**。
- `role: "assistant"`：模型的回复。这是 DPO 算法主要学习差异的地方。

接下来我们看看DPODataset类

```python
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取 padding token id，如果没有则默认为 0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # =================================================================
        # 核心逻辑：定义“助手回答开始”和“助手回答结束”的标志 Token
        # =================================================================
        # 这段代码假设了一种特定的 Chat Template 格式（例如 MiniMind 或 Llama 3 格式）
        # bos_id 这里特指 Assistant 回答的起始特征，例如 "<bos>assistant\n"
        # 只有检测到这个序列，才意味着接下来的内容是模型需要学习的回答
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        
        # eos_id 特指回答结束的特征，例如 "<eos>\n"
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        
        # 使用 HuggingFace datasets 库加载 json 格式的数据文件
        self.data = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        # 返回数据集的总样本数
        return len(self.data)

    def __getitem__(self, index):
        # 获取单条数据
        item = self.data[index]
        
        # 获取 chosen (好回答) 和 rejected (坏回答) 的对话列表
        # 格式通常是: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        chosen = item['chosen']
        rejected = item['rejected']
        
        # =======================================================
        # 1. 应用对话模板 (Apply Chat Template)
        # =======================================================
        # 将 list 格式的对话转换成纯文本字符串。
        # tokenize=False 表示只拼接字符串，暂不转成 token id，方便后续统一处理
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # =======================================================
        # 2. Tokenizer 编码
        # =======================================================
        # 将文本转换为 token id 序列。
        # truncation=True: 超过 max_length 截断
        # padding='max_length': 不足 max_length 补 0 (padding token)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 提取 input_ids (token 序列)
        chosen_input_ids = chosen_encoding['input_ids']
        # 生成 mask (用于标记哪些 token 是助手的回答，需要计算 loss)
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # =======================================================
        # 3. 构造训练数据 (Shift Trick)
        # =======================================================
        # 在因果语言模型（Causal LM）训练中，我们预测下一个 token。
        # 输入是 x (0 到 N-1)，目标是 y (1 到 N)。
        
        # chosen 数据的输入 (去掉最后一个 token)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        # chosen 数据的标签 (去掉第一个 token，向左位移一位)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        # mask 也要对应 y 的位置，所以同样去掉第一个
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        # rejected 数据同理
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """
        核心函数：生成 Loss Mask
        目标：只保留 Assistant 回答部分的 Loss，忽略 User 输入和 Padding。
        """
        # 初始化全 0 的 mask (默认都不计算 loss)
        loss_mask = [0] * len(input_ids)
        i = 0
        
        # 遍历整个 token 序列
        while i < len(input_ids):
            # 1. 寻找 Assistant 回答的“开始标记” (例如 "assistant\n")
            # 检查当前位置 i 是否匹配 bos_id 序列
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 找到了开始标记，start 指向回答内容的第一个 token
                start = i + len(self.bos_id)
                end = start
                
                # 2. 寻找 Assistant 回答的“结束标记” (例如 "<eos>\n")
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 3. 将 Start 到 End 之间的部分 mask 设为 1
                # 这部分就是模型实际生成的回答，我们需要计算它的 Loss
                # min(..., max_length) 防止越界
                # end + len(self.eos_id) 是为了让模型学会生成结束符 (EOS) 本身
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                
                # 更新 i 指针，跳过这段已经处理完的回答，继续寻找下一轮对话（如果是多轮对话）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 如果没匹配到开始标记，指针后移一位继续找
                i += 1
                
        return loss_mask
```

我们输出一条处理后的数据，看看是什么样的

```python
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_root, "model"))

    dpo_path = os.path.join(_root, "dataset", "dpo.jsonl")
    if not os.path.isfile(dpo_path):
        print("未找到 dpo.jsonl:", dpo_path)
        raise SystemExit(1)

    ds = DPODataset(dpo_path, tokenizer, max_length=1024)
    if len(ds) == 0:
        print("DPO 数据集为空")
        raise SystemExit(1)

    for i in range(min(1, len(ds))):
        out = ds[i]
        print("=" * 60)
        print("--- Sample {} ---".format(i))

        # ---------- chosen ----------
        print("\n[chosen]")
        x_chosen = out["x_chosen"]
        mask_chosen = out["mask_chosen"]
        chosen_ids = x_chosen.tolist()
        chosen_len = len(chosen_ids)
        if tokenizer.pad_token_id is not None:
            while chosen_len > 0 and chosen_ids[chosen_len - 1] == tokenizer.pad_token_id:
                chosen_len -= 1
        print("解码前 (input_ids, 去 pad 后长度 {}):".format(chosen_len))
        print(chosen_ids[:chosen_len])
        print("解码后 (文本):")
        print(tokenizer.decode(chosen_ids[:chosen_len], skip_special_tokens=False))
        print("mask_chosen (0=不计算 loss, 1=计算, 长度 {}):".format(mask_chosen.shape[0]))
        print(mask_chosen.tolist())
        print("mask_chosen 非零个数:", mask_chosen.sum().item())

        # ---------- rejected ----------
        print("\n[rejected]")
        x_rejected = out["x_rejected"]
        mask_rejected = out["mask_rejected"]
        rej_ids = x_rejected.tolist()
        rej_len = len(rej_ids)
        if tokenizer.pad_token_id is not None:
            while rej_len > 0 and rej_ids[rej_len - 1] == tokenizer.pad_token_id:
                rej_len -= 1
        print("解码前 (input_ids, 去 pad 后长度 {}):".format(rej_len))
        print(rej_ids[:rej_len])
        print("解码后 (文本):")
        print(tokenizer.decode(rej_ids[:rej_len], skip_special_tokens=False))
        print("mask_rejected (0=不计算 loss, 1=计算, 长度 {}):".format(mask_rejected.shape[0]))
        print(mask_rejected.tolist())
        print("mask_rejected 非零个数:", mask_rejected.sum().item())
        print()

```

```bash
(myminimind) root@GPU04:~/minimind/dataset$ python lm_dataset.py 
/home/zmm/anaconda3/envs/myminimind/lib/python3.10/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
============================================================
--- Sample 0 ---

[chosen]
解码前 (input_ids, 去 pad 后长度 307):
[1, 85, 736, 201, 59, 292, 389, 260, 3836, 1861, 501, 2, 201, 1, 320, 275, 201, 1539, 86, 261, 1007, 2, 201, 1, 1078, 538, 501, 201, 4807, 2457, 352, 305, 449, 738, 366, 665, 85, 2004, 85, 14, 670, 2061, 290, 606, 660, 1174, 2337, 427, 3817, 4001, 968, 1265, 261, 3547, 281, 5693, 3519, 2353, 16, 3467, 761, 1357, 565, 429, 935, 2993, 14, 670, 1036, 4969, 426, 1071, 860, 3693, 349, 296, 1802, 357, 3880, 281, 4882, 4205, 2337, 4256, 16, 2422, 1036, 804, 3943, 5153, 823, 2558, 281, 3252, 431, 78, 4901, 2759, 2664, 290, 1160, 5153, 2337, 4256, 16, 201, 201, 1701, 4240, 14, 670, 5649, 290, 2255, 1091, 1174, 2099, 665, 1942, 1180, 290, 272, 1056, 290, 1337, 5472, 14, 751, 432, 271, 782, 3218, 2099, 665, 85, 14, 1478, 1253, 2099, 665, 85, 374, 5690, 85, 14, 281, 6387, 787, 3379, 1161, 1829, 16, 2422, 1036, 1752, 77, 290, 1182, 1845, 427, 4389, 503, 278, 1361, 4153, 2514, 1365, 574, 1636, 89, 558, 290, 846, 260, 3124, 2183, 303, 2099, 665, 2417, 16, 201, 201, 4918, 1036, 804, 606, 660, 1174, 5693, 2628, 290, 5577, 260, 288, 1795, 4847, 14, 5992, 334, 2797, 297, 3830, 606, 635, 433, 1407, 3759, 278, 4414, 5919, 2621, 14, 1095, 852, 5693, 14, 281, 3257, 2901, 16, 2422, 1036, 4753, 786, 369, 1561, 460, 290, 2072, 683, 1174, 2555, 2181, 398, 940, 1018, 2608, 281, 1272, 2338, 4666, 290, 2611, 1174, 1942, 1180, 290, 276, 5375, 15, 352, 855, 282, 2315, 281, 4007, 303, 1174, 3885, 2643, 16, 201, 201, 4807, 670, 1097, 3229, 14, 670, 1036, 5940, 4670, 2975, 290, 1254, 3338, 282, 1174, 4595, 299, 290, 789, 1652, 275, 281, 2496, 6130, 1407, 4455, 1189, 2099, 665, 2417, 349, 3943, 1236, 2993, 14, 1478, 1253, 14, 281, 6291, 16, 2, 201]
解码后 (文本):
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
continue<|im_end|>
<|im_start|>assistant
As Recharge Retreats grows, we plan to expand our team with additional event coordinators and marketing specialists. To accommodate this growth, we will establish an office space that fosters collaboration and creativity among team members. We will also prioritize remote work options and streamlined communication tools to support remote team members.

In addition, we aim to diversify our retreat offerings to cater to different themes, such as solo travel retreats, wellness retreats for parents, and spiritual journeys. We will seek to partner with multiple venues across various locations worldwide to provide a diverse range of retreat experiences.

We will also expand our marketing efforts to reach a wider audience, maximizing brand exposure through influencer partnerships, experiential marketing, and virtual events. We will continuously strive to refine our services based on client feedback and identify industry trends to adapt our offerings to the ever-changing needs and preferences of our target market.

As we scale, we will remain committed to upholding our mission to empower and connect women through transformative retreat experiences that prioritize personal growth, wellness, and connection.<|im_end|>

mask_chosen (0=不计算 loss, 1=计算, 长度 1023):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mask_chosen 非零个数: 279

[rejected]
解码前 (input_ids, 去 pad 后长度 358):
[1, 85, 736, 201, 59, 292, 389, 260, 3836, 1861, 501, 2, 201, 1, 320, 275, 201, 1539, 86, 261, 1007, 2, 201, 1, 1078, 538, 501, 201, 35, 16, 4826, 4863, 28, 201, 201, 19, 16, 982, 563, 275, 260, 6211, 280, 352, 1986, 2022, 28, 2936, 890, 2476, 290, 3375, 281, 2850, 429, 2457, 352, 305, 449, 738, 366, 665, 85, 6211, 280, 352, 5254, 295, 1337, 6112, 14, 606, 660, 282, 1174, 5577, 281, 2205, 278, 1007, 1028, 16, 201, 20, 16, 4114, 2751, 6277, 281, 968, 4288, 2640, 28, 489, 263, 429, 3617, 1922, 1185, 604, 529, 3252, 431, 85, 645, 6270, 2277, 281, 1160, 290, 1695, 281, 2484, 4930, 295, 1912, 282, 546, 2018, 2099, 665, 85, 16, 201, 201, 36, 16, 720, 1424, 4277, 28, 201, 201, 19, 16, 3206, 82, 660, 2565, 1942, 1180, 28, 4367, 1795, 1155, 282, 3817, 2099, 665, 15, 4964, 871, 2555, 14, 751, 432, 968, 4288, 14, 6144, 1160, 14, 2099, 665, 4401, 337, 14, 437, 2099, 665, 2915, 1562, 2558, 16, 201, 20, 16, 350, 3620, 1017, 2195, 1943, 28, 599, 306, 480, 4208, 1174, 3885, 2643, 290, 1819, 765, 2708, 4404, 1238, 14, 751, 432, 5501, 14, 5249, 440, 14, 437, 1635, 2101, 3353, 16, 201, 201, 37, 16, 1088, 2701, 2029, 3206, 82, 662, 299, 28, 201, 201, 19, 16, 614, 712, 1845, 427, 2058, 2484, 28, 5984, 429, 427, 6130, 531, 2484, 634, 306, 480, 290, 2061, 6096, 2099, 665, 85, 281, 606, 660, 1174, 4420, 3484, 743, 16, 201, 20, 16, 487, 2546, 77, 4689, 282, 2582, 28, 614, 373, 85, 1007, 2586, 1551, 14, 2684, 994, 14, 281, 649, 662, 290, 2046, 697, 6096, 2099, 665, 85, 281, 1160, 606, 662, 299, 295, 1017, 2195, 1943, 16, 201, 201, 3771, 3428, 935, 1838, 2061, 14, 2457, 352, 305, 449, 738, 366, 665, 85, 780, 276, 1028, 290, 2433, 260, 2948, 1123, 275, 303, 6130, 531, 2099, 665, 2555, 14, 789, 1652, 2550, 1695, 281, 2484, 290, 296, 6000, 2993, 14, 6291, 14, 281, 1236, 4455, 337, 1407, 2185, 279, 643, 281, 1524, 15, 278, 74, 5513, 2417, 16, 2, 201]
解码后 (文本):
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
continue<|im_end|>
<|im_start|>assistant
A. Scaling:

1. Offer a franchise model: Allow others to open and operate Recharge Retreats franchises in different regions, expanding our reach and revenue potential.
2. Develop online courses and coaching programs: Create supplementary income streams by offering training and support to individuals and organizations interested in organizing their own retreats.

B. Diversification:

1. Expand service offerings: Consider adding additional retreat-related services, such as coaching, ongoing support, retreat evaluation, or retreat financing options.
2. Target new markets: Broadening our target market to include other demographics, such as men, families, or specific interest groups.

C. International Expansion:

1. Partner with local organizations: Collaborate with women's organizations abroad to plan international retreats and expand our global presence.
2. Seek funding opportunities: Pursue grants, investments, and loans to finance international retreats and support expansion in new markets.

By following this business plan, Recharge Retreats has the potential to become a leading provider of women's retreat services, empowering individuals and organizations to foster growth, connection, and personal transformation through memorable and life-enhancing experiences.<|im_end|>

mask_rejected (0=不计算 loss, 1=计算, 长度 1023):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mask_rejected 非零个数: 330
```

读者可以仔细看看数据是怎么被处理的。

每条样本包含：

- chosen

  - 解码前：x_chosen 的 token id 列表（已去掉尾部 padding），并注明去 pad 后的长度

  - 解码后：上述 id 用 tokenizer.decode 得到的完整文本

  - mask_chosen：整段序列的 0/1 列表（0=不参与 loss，1=参与）及非零个数

- rejected

  - 解码前：x_rejected 的 token id 列表（已去掉尾部 padding）及长度

  - 解码后：对应解码文本

  - mask_rejected：整段 0/1 mask 列表及非零个数

## 2. 核心训练引擎：`train_DPO.py`

我们先来看看这份代码与之前的几个训练代码不一样的地方

```python
def logits_to_log_probs(logits, labels):
    """
    工具函数：将模型的原始输出 logits 转换为对应 label 的 log 概率。
    这对于计算 DPO Loss 至关重要，因为我们需要知道模型生成特定 token 的概率。
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len, vocab_size)
    
    # 1. 对 logits 进行 log_softmax 归一化，得到所有词汇的 log 概率
    log_probs = F.log_softmax(logits, dim=2)
    
    # 2. gather 操作：只提取 labels 对应位置的那个 token 的概率
    # index=labels.unsqueeze(2) 将 labels 维度变为 (batch, seq, 1) 以匹配 gather 需求
    # .squeeze(-1) 将结果变回 (batch, seq)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    
    return log_probs_per_token  # shape: (batch_size, seq_len)
```



```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    DPO Loss 的核心计算函数。
    公式: Loss = -log σ(β * (log(π_theta(yw)/π_ref(yw)) - log(π_theta(yl)/π_ref(yl))))
    """
    # ref_log_probs 和 policy_log_probs 形状均为: (batch_size, seq_len)
    # 这里的 batch_size 实际上包含了 chosen 和 rejected 拼接后的数量 (即 2 * 真实 batch_size)

    # 1. 计算每个样本的有效长度（排除 padding）
    # clamp_min(1e-8) 是为了防止全 padding 的异常数据导致除以 0
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8) # mask全是1或者0，那么加起来就可以得到有效seq_lengths
    
    # 2. 计算整个句子的平均 log 概率
    # 乘以 mask 是为了把 padding 部分的概率置为 0，然后求和并除以有效长度
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 3. 将拼接在一起的数据拆分回 chosen (好回答) 和 rejected (坏回答)
    # 假设输入 batch 是 [chosen_1, chosen_2, ..., rejected_1, rejected_2, ...]
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 4. 计算策略模型对 好回答 和 坏回答 的 log 概率差
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    
    # 5. 计算参考模型对 好回答 和 坏回答 的 log 概率差 (这是基准)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # 6. 计算 DPO 的核心 logits
    # 含义：策略模型相对于参考模型，是否更偏向于“好回答”
    logits = pi_logratios - ref_logratios
    
    # 7. 计算损失：使用 log sigmoid
    # beta 是超参数，控制由于偏离参考模型带来的惩罚力度
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()
```

### 第一步：计算每个样本的有效长度

mask全是1或者0，那么加起来就可以得到有效seq_lengths

### 第二步：计算整个句子的平均 log 概率

这里的 `.sum(dim=1)` 在数学上执行的是 **从“单个词的概率”到“整句话的概率”的转换**。

在概率论中，计算一个序列（一句话）的生成概率，是将其中每个词（Token）的概率**相乘**。但是因为我们是在 **对数空间 (Log Space)** 操作，乘法就变成了**加法**。

具体步骤如下：

#### 1. 数学原理：乘法变加法

假设一句话只有三个词 "I", "love", "AI"。

模型生成这句话的概率 $P(\text{sentence})$ 是每个词概率的乘积：

$$P(\text{sentence}) = P(\text{"I"}) \times P(\text{"love"}) \times P(\text{"AI"})$$

如果我们对两边取对数（Log），乘号就变成了加号：

$$\log P(\text{sentence}) = \log P(\text{"I"}) + \log P(\text{"love"}) + \log P(\text{"AI"})$$

这就是 `sum(dim=1)` 在做的事情：**它把这句话中所有有效 Token 的 Log 概率加起来，得到整句话的总 Log 概率。**

------

#### 2. 代码执行流程

假设 `batch_size=1`，一句话只有 4 个位置。

句子是：`[User: Hi] [Assistant: Hello] [Pad]`

我们只关心 Assistant 的回答 "Hello"。

##### **Step 1: 原始输入**

- **`log_probs`**: 模型对每个位置预测正确 Token 的 Log 概率（通常是负数，越接近 0 越好）。
- **`mask`**: 标记哪些是 Assistant 的回答（1 是回答，0 是用户提问或 Pad）。

| **位置** | **Token** | **mask** | **log_probs (假设值)** | **含义**                   |
| -------- | --------- | -------- | ---------------------- | -------------------------- |
| 0        | User      | 0        | -5.2                   | (用户提问，不关心)         |
| 1        | :         | 0        | -0.1                   | (分隔符，不关心)           |
| 2        | **Hello** | **1**    | **-0.5**               | **(Assistant 回答，重要)** |
| 3        | Pad       | 0        | -0.0                   | (填充，不关心)             |

##### **Step 2: Mask 过滤 (`log_probs \* mask`)**

代码执行 `log_probs * mask`，将不需要计算 Loss 的部分置为 0。

| **位置** | **计算过程**  | **结果** |
| -------- | ------------- | -------- |
| 0        | -5.2 * 0      | 0.0      |
| 1        | -0.1 * 0      | 0.0      |
| 2        | **-0.5 \* 1** | **-0.5** |
| 3        | -0.0 * 0      | 0.0      |

##### **Step 3: 求和 (`.sum(dim=1)`)**

代码执行 `.sum(dim=1)`，沿着序列长度方向（横向）把所有数字加起来。

$$0.0 + 0.0 + (-0.5) + 0.0 = \mathbf{-0.5}$$

这个 **-0.5** 就是模型生成 "Hello" 这个回答的总 Log 概率。

------

#### 3. 为什么要除以 `seq_lengths`？

注意代码的下一部分：`/ seq_lengths.squeeze()`。

虽然 `sum` 算出了整句话的概率，但**长句子天然比短句子的 Sum 值更小**（因为 Log 概率是负数，加得越多越负）。

- 短句 "Hi" 的 Log Sum 可能是 -0.5
- 长句 "I am very happy today" 的 Log Sum 可能是 -5.0

如果不做归一化，DPO 可能会错误地认为短句子总是比长句子“概率更高/更好”。

因此，代码最后除以了句子的有效长度，计算的是 **“平均每个 Token 的 Log 概率”**，这样长短句就可以公平比较了。















