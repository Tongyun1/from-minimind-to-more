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
    # eval_tokenizer()

if __name__ == "__main__":
    # 当脚本直接运行时（而非被导入），执行主函数
    main()