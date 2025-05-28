from collections import defaultdict

class FastBPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def build_initial_corpus(self, words):
        corpus = [list(word) for word in words]
        print(f"[Initial Corpus] {corpus}")
        return corpus

    def get_initial_stats(self, corpus):
        pair_freq = defaultdict(int)
        pair_positions = defaultdict(set)
        for i, word in enumerate(corpus):
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_freq[pair] += 1
                pair_positions[pair].add((i, j))
        print(f"[Initial Stats] Pair Frequencies: {dict(pair_freq)}")
        print(f"[Initial Pair Positions] {dict(pair_positions)}")
        return pair_freq, pair_positions

    # 更新 pair_freq 和 pair_positions，完整取代原來的版本
    # ⚠️ 正確處理「消失或移位的舊 pair」：先刪後補

    def merge_pair(self, pair, corpus, pair_positions, pair_freq):
        replacement = ''.join(pair)
        affected_positions = pair_positions[pair].copy()
        new_pair_positions = defaultdict(set)
        # new_pair_freq = defaultdict(int)

        print(f"\n[Merging Pair] {pair} -> '{replacement}'")
        print(f"[Affected Positions] {affected_positions}")

        affected_indices = set(i for i, _ in affected_positions)

        # print(f"before discard: {pair_positions}")
        # print(f"before discard: {pair_freq}")
        # 💡 先清掉受影響詞中所有舊 pair 的位置
        for i in affected_indices:
            word = corpus[i]
            original_word_len = len(word)
            for j in range(original_word_len - 1):
                old_pair = (word[j], word[j + 1])
                print(f"old_pair: {old_pair}")
                if old_pair in pair_positions:
                    pair_positions[old_pair].discard((i, j))
                    # print(f"old_pair, i, j: {old_pair}, {i}, {j}")
                    if not pair_positions[old_pair]:
                        del pair_positions[old_pair]

                if old_pair in pair_freq:
                    pair_freq[old_pair] -= 1
                    if pair_freq[old_pair] <= 0:
                        del pair_freq[old_pair]

        # print(f"after discard: {pair_positions}")
        # print(f"after discard: {pair_freq}")

        for i, j in affected_positions:
            word = corpus[i]
            if j >= len(word) - 1 or (word[j], word[j + 1]) != pair:
                continue
            new_word = word[:j] + [replacement] + word[j + 2:]
            corpus[i] = new_word

        # 再重建新的 pair stats
        for i in affected_indices:
            word = corpus[i]
            for j in range(len(word) - 1):
                new_pair = (word[j], word[j + 1])
                new_pair_positions[new_pair].add((i, j))
                pair_freq[new_pair] += 1

        print(f"[Corpus After Merge] {corpus}")
        print(f"[Updated Pair Positions from Affected Words] {dict(new_pair_positions)}")
        return corpus, pair_freq, new_pair_positions

    def train(self, words):
        corpus = self.build_initial_corpus(words)
        pair_freq, pair_positions = self.get_initial_stats(corpus)

        merges_done = 0
        while merges_done < self.vocab_size and pair_freq:
            best_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
            print(f"\n[Step {merges_done + 1}] Best Pair: {best_pair}, Frequency: {pair_freq[best_pair]}")

            self.merges.append(best_pair)
            corpus, pair_freq, new_positions = self.merge_pair(best_pair, corpus, pair_positions, pair_freq)

            # 刪除剛合併的 pair
            # print(f"[Before Deletion] pair_positions has {len(pair_positions)} entries")
            if best_pair in pair_freq:
                del pair_freq[best_pair]
            if best_pair in pair_positions:
                del pair_positions[best_pair]
            # print(f"[After Deletion] pair_positions has {len(pair_positions)} entries")

            # 更新已受影響詞的 pair 統計
            # for pair, freq in new_freq.items():
            #     pair_freq[pair] = freq
            for pair, positions in new_positions.items():
                pair_positions[pair] = positions

            print(f"[Full Updated Pair Freq] {dict(pair_freq)}")
            print(f"[Full Updated Pair Positions] {dict(pair_positions)}")
            merges_done += 1

        self.vocab = {"".join(word): i for i, word in enumerate(corpus)}
        print(f"\n[Final Vocabulary] {self.vocab}")
        print(f"[Merge History] {self.merges}")
        return self.vocab, self.merges

# 測試用
corpus = ["low", "lower", "newest", "widest"]
bpe = FastBPE(vocab_size=10)
vocab, merges = bpe.train(corpus)
