from collections import defaultdict

class NaiveBPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = []

    def build_initial_corpus(self, words):
        corpus = [list(word) for word in words]
        print(f"[Initial Corpus] {corpus}")
        return corpus

    def get_pair_frequencies(self, corpus):
        pair_freq = defaultdict(int)
        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += 1
        return pair_freq

    def merge_pair(self, pair, corpus):
        merged_token = ''.join(pair)
        new_corpus = []

        for word in corpus:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(new_word)

        return new_corpus

    def train(self, words):
        corpus = self.build_initial_corpus(words)

        for step in range(self.vocab_size):
            pair_freq = self.get_pair_frequencies(corpus)
            if not pair_freq:
                break

            best_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
            print(f"\n[Step {step + 1}] Merging pair: {best_pair}, Frequency: {pair_freq[best_pair]}")

            self.merges.append(best_pair)
            corpus = self.merge_pair(best_pair, corpus)
            print(f"[Corpus After Merge] {corpus}")

        vocab = {"".join(word): i for i, word in enumerate(corpus)}
        print(f"\n[Final Vocabulary] {vocab}")
        print(f"[Merge History] {self.merges}")
        return vocab, self.merges

# 測試用
corpus = ["low", "lower", "newest", "widest"]
bpe = NaiveBPE(vocab_size=10)
vocab, merges = bpe.train(corpus)
