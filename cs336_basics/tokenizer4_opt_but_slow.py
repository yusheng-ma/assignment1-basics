import regex as re
from collections import defaultdict


# ==== Pair Frequency Counting ====
def get_stats(vocab):
    pairs = defaultdict(int)
    for byte_word, freq in vocab.items():
        for pair in zip(byte_word, byte_word[1:]):
            pairs[pair] += freq
    return pairs


# ==== Initial Bigram Stats ====
def get_initial_stats(byte_corpus):
    pair_freq = defaultdict(int)
    for word in byte_corpus:
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            pair_freq[pair] += 1
    return pair_freq


# ==== Merge Bigram in Vocabulary ====
def merge_pair(pair, byte_vocab, pair_freq, pair_to_byte_vocab):
    replacement = pair[0] + pair[1]
    words = list(pair_to_byte_vocab[pair])

    for word in words:
        freq = byte_vocab[word]
        del byte_vocab[word]

        # Remove old pairs
        for j in range(len(word) - 1):
            old_pair = (word[j], word[j + 1])
            pair_freq[old_pair] -= freq
            if pair_freq[old_pair] <= 0:
                pair_freq.pop(old_pair, None)
            pair_to_byte_vocab[old_pair].discard(word)
            if not pair_to_byte_vocab[old_pair]:
                del pair_to_byte_vocab[old_pair]

        # Merge word
        w_out = []
        j = 0
        while j < len(word):
            if j < len(word) - 1 and (word[j], word[j + 1]) == pair:
                w_out.append(replacement)
                j += 2
            else:
                w_out.append(word[j])
                j += 1

        new_word = tuple(w_out)
        byte_vocab[new_word] += freq

        # Add new pairs
        for j in range(len(new_word) - 1):
            new_pair = (new_word[j], new_word[j + 1])
            pair_freq[new_pair] += freq
            pair_to_byte_vocab[new_pair].add(new_word)

    return byte_vocab, pair_freq, pair_to_byte_vocab


# ==== BPE Trainer ====
def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # Add special tokens
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token.encode("utf-8")

    # Read and tokenize input
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    split_pattern = "|".join(map(re.escape, special_tokens))
    chunks = re.split(split_pattern, content)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    byte_vocab = defaultdict(int)
    byte_corpus = []

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            byte_seq = [bytes([i]) for i in match.group().encode("utf-8")]
            token = tuple(byte_seq)
            byte_vocab[token] += 1
            byte_corpus.append(byte_seq)

    pair_freq = get_initial_stats(byte_corpus)

    pair_to_byte_vocab = defaultdict(set)
    for word in byte_vocab:
        for pair in zip(word, word[1:]):
            pair_to_byte_vocab[pair].add(word)

    # Run merges
    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not pair_freq:
            break
        best_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        byte_vocab, pair_freq, pair_to_byte_vocab = merge_pair(
            best_pair, byte_vocab, pair_freq, pair_to_byte_vocab
        )
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

    return vocab, merges


# ==== Main Entry ====
def main():
    input_path = "/mnt/disk3/yusheng/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    bpe_train(input_path, vocab_size, special_tokens)


if __name__ == "__main__":
    main()
