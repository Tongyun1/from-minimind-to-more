class BPETokenizer:
    def __init__(self):
        # 记录合并规则：字典保持插入顺序 (Python 3.7+)，
        # 越早插入的规则优先级越高（Rank 越靠前）
        self.merges = {} 

    def _get_stats(self, tokens):
        """内部方法：统计相邻 token 对的出现频率"""
        counts = {}
        # zip(tokens, tokens[1:]) 是一种优雅的获取所有相邻元素对的方法
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, tokens, pair, new_token):
        """内部方法：在 tokens 列表中，将指定的 pair 替换为 new_token"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            # 如果匹配到了这对组合，并且不是最后一个字符
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(new_token)
                i += 2 # 跳过这两个已经被合并的 token
            else:
                new_tokens.append(tokens[i])
                i += 1 # 仅移动一步
        return new_tokens

    def train(self, text, num_merges):
        """训练分词器：从语料中学习合并规则"""
        # 初始状态：将文本拆分为单个字符（真实的BPE通常会将其转为UTF-8字节）
        tokens = list(text)
        print(f"【训练开始】初始字符数: {len(tokens)}")

        for i in range(num_merges):
            counts = self._get_stats(tokens)
            if not counts:
                break # 没有可以合并的对儿了

            # 找到当前出现频率最高的相邻 token 对
            best_pair = max(counts, key=counts.get)
            new_token = best_pair[0] + best_pair[1]

            # 记录合并规则
            self.merges[best_pair] = new_token

            # 执行合并并更新 tokens 列表
            tokens = self._merge(tokens, best_pair, new_token)
            print(f"合并步骤 {i+1}: {best_pair[0]:<4} + {best_pair[1]:<4} -> '{new_token}' (频率: {counts[best_pair]})")
            
        print(f"【训练结束】最终 Token 数: {len(tokens)}\n")

    def tokenize(self, text):
        """使用学习到的规则对新文本进行分词"""
        tokens = list(text)

        while len(tokens) > 1:
            pairs = self._get_stats(tokens)
            
            # 找到在我们的训练规则 (self.merges) 中存在的 pair
            # 因为字典保持了插入顺序，所以我们遍历时先遇到的一定是
            # 训练时最早被合并的 pair（即优先级最高的 pair）
            pair_to_merge = None
            for pair in self.merges:
                if pair in pairs:
                    pair_to_merge = pair
                    break

            if not pair_to_merge:
                break # 如果文本中的所有对都不在我们的规则里，就停止合并

            # 执行合并
            new_token = self.merges[pair_to_merge]
            tokens = self._merge(tokens, pair_to_merge, new_token)

        return tokens


# ==========================================
# 测试与运行示例
# ==========================================
if __name__ == "__main__":
    # 1. 实例化分词器
    tokenizer = BPETokenizer()

    # 2. 准备训练语料 (故意使用一些重复出现的模式，如 "ab" 和 "abac")
    training_corpus = "abac abac dabac aabac"
    
    # 3. 训练分词器，指定执行 4 次合并操作
    tokenizer.train(training_corpus, num_merges=4)

    # 4. 打印学习到的合并规则
    print("【学习到的合并规则字典】:")
    for pair, new_token in tokenizer.merges.items():
        print(f"{pair} -> {new_token}")
    print("\n")

    # 5. 测试分词器 (对训练集内的文本进行分词)
    test_text_1 = "dabac"
    print(f"分词测试 1 ('{test_text_1}'): {tokenizer.tokenize(test_text_1)}")

    # 6. 测试分词器 (对未见过的，但包含已学习模式的文本分词)
    test_text_2 = "zabacy"
    print(f"分词测试 2 ('{test_text_2}'): {tokenizer.tokenize(test_text_2)}")