import regex as re
from collections import defaultdict


# ==== Pair Counting from Vocabulary ====
def get_stats(vocab):
    pairs = defaultdict(int)
    for byte_word, freq in vocab.items():
        for pair in zip(byte_word, byte_word[1:]):
            pairs[pair] += freq
    return pairs


# ==== Initial Pair Frequency and Positions ====
def get_initial_stats(byte_corpus):
    pair_freq = defaultdict(int)
    pair_positions = defaultdict(set)
    for i, word in enumerate(byte_corpus):
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            pair_freq[pair] += 1
            pair_positions[pair].add((i, j))
    return pair_freq, pair_positions


# ==== Merge a Single Pair in Vocabulary ====
def merge_vocab(pair, v_in):
    v_out = {}
    for word, freq in v_in.items():
        w_out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                w_out.append(word[i] + word[i + 1])
                i += 2
            else:
                w_out.append(word[i])
                i += 1
        v_out[tuple(w_out)] = freq
    return v_out


# ==== Apply Merge to Corpus and Update Stats ====
def merge_pair(pair, corpus, pair_freq, pair_positions, byte_vocab, pair_to_byte_vocab):
    replacement = pair[0] + pair[1]
    affected_positions = pair_positions[pair].copy()
    affected_indices = set(i for i, _ in affected_positions)
    
    # operate on each word
    words = list(pair_to_byte_vocab[pair])
    for word in words:
        freq = byte_vocab[word]
        byte_vocab[word] -= freq
        # remove old byte_vocab
        if byte_vocab[word] <= 0:
            del byte_vocab[word]
        # remove old pair_freq and pair_to_byte_vocab
        for j in range(len(word) - 1):
            old_pair = (word[j], word[j + 1])
            pair_freq[old_pair] -= freq
            if pair_freq[old_pair] <= 0:
                del pair_freq[old_pair]
            if word in pair_to_byte_vocab[old_pair]:
                pair_to_byte_vocab[old_pair].discard(word)
                if not pair_to_byte_vocab[old_pair]:
                    del pair_to_byte_vocab[old_pair]
        # actually merge
        w_out = []
        j = 0
        while j < len(word):
            if j < len(word) - 1 and (word[j], word[j + 1]) == pair:
                w_out.append(replacement)
                j += 2
            else:
                w_out.append(word[j])
                j += 1
        # add new byte_vocab
        # add new pair_freq and pair_to_byte_vocab
        byte_vocab[tuple(w_out)] += freq
        for new_pair in zip(w_out, w_out[1:]):
            pair_freq[new_pair] += freq
            pair_to_byte_vocab[new_pair].add(tuple(w_out))

    return corpus, pair_freq, pair_positions, byte_vocab, pair_to_byte_vocab


# ==== Main BPE Training Loop ====
def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # Add special tokens
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for i, token_bytes in enumerate(special_token_bytes, start=len(vocab)):
        vocab[i] = token_bytes

    # Read file and pre-tokenize
    with open(input_path, "r") as f:
        content = f.read()

    split_pattern = "|".join(map(re.escape, special_tokens))
    chunks = re.split(split_pattern, content)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    byte_vocab = defaultdict(int)
    byte_corpus = []
    for content in chunks:
        for match in re.finditer(PAT, content):
            byte_seq = [bytes([i]) for i in match.group().encode('utf-8')]
            byte_vocab[tuple(byte_seq)] += 1
            byte_corpus.append(byte_seq)

    # Initialize pair stats
    pair_freq, pair_positions = get_initial_stats(byte_corpus)
    pair_to_byte_vocab = defaultdict(set)
    for word in byte_vocab:
        for pair in zip(word, word[1:]):
            pair_to_byte_vocab[pair].add(word)

    # Train BPE merges
    num_merges = vocab_size - len(vocab)
    for index in range(len(vocab), len(vocab) + num_merges):
        if not pair_freq:
            break
        best = max(pair_freq, key=lambda k: (pair_freq[k], k))
        byte_corpus, pair_freq, pair_positions, byte_vocab, pair_to_byte_vocab = merge_pair(
            best, byte_corpus, pair_freq, pair_positions, byte_vocab, pair_to_byte_vocab
        )
        vocab[index] = best[0] + best[1]
        merges.append(best)

    return vocab, merges


# ==== Entrypoint ====
def main():
    input_path = "/mnt/disk3/yusheng/assignment1-basics/data/my_test2.txt"
    vocab_size = 256 + 10
    special_tokens = ["<|endoftext|>"]
    bpe_train(input_path, vocab_size, special_tokens)


if __name__ == "__main__":
    main()
