import regex as re
from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for byte_word, freq in vocab.items():
        for pair in zip(byte_word, byte_word[1:]):
            pairs[pair] += freq
    return pairs

# [optimization]
def get_initial_stats(byte_corpus):
    pair_freq = defaultdict(int)
    pair_positions = defaultdict(set)
    for i, word in enumerate(byte_corpus):
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            pair_freq[pair] += 1
            pair_positions[pair].add((i, j))
    # print(f"[Initial Stats] Pair Frequencies: {dict(pair_freq)}")
    # print(f"[Initial Pair Positions] {dict(pair_positions)}")
    return pair_freq, pair_positions

def merge_vocab(pair, v_in):
    v_out = {}
    for word, freq in v_in.items():
        w_out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                w_out.append(word[i] + word[i+1]) # concat
                i += 2
            else:
                w_out.append(word[i])
                i += 1
        v_out[tuple(w_out)] = freq

    return v_out

# [optimization]
def merge_pair(pair, corpus, pair_freq, pair_positions):
    replacement = pair[0] + pair[1]
    # print(pair)
    # print(pair_positions)
    # print(pair_positions[pair])
    affected_positions = pair_positions[pair].copy() # deep copy otherwise get removed old
    affected_indices = set(i for i, _ in affected_positions)

    # remove old
    for i in affected_indices:
        word = corpus[i]
        original_word_len = len(word)
        for j in range(original_word_len - 1):
            old_pair = (word[j], word[j + 1])
            if old_pair in pair_freq:
                pair_freq[old_pair] -= 1
                if pair_freq[old_pair] <= 0:
                    del pair_freq[old_pair]
            if old_pair in pair_positions:
                pair_positions[old_pair].discard((i, j))
                if not pair_positions[old_pair]:
                    del pair_positions[old_pair]
            
    # merge corpus
    for i, j in affected_positions:
        word = corpus[i]
        if j >= len(word) - 1 or (word[j], word[j + 1]) != pair:
            continue
        new_word = word[:j] + [replacement] + word[j + 2:]
        corpus[i] = new_word

    # add new
    for i in affected_indices:
        word = corpus[i]
        for j in range(len(word) - 1):
            new_pair = (word[j], word[j + 1])
            pair_freq[new_pair] += 1
            pair_positions[new_pair].add((i, j))

    # remove best
    if pair in pair_freq:
        del pair_freq[pair]
    if pair in pair_positions:
        del pair_positions[pair]
    
    return corpus, pair_freq, pair_positions
    
def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # init vocab and merges
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # add special tokens to vocab
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for i, token_bytes in enumerate(special_token_bytes, start=len(vocab)):
        vocab[i] = token_bytes
    
    # read file
    with open(input_path, "r") as f:
        content = f.read()
    # print(content)
    
    # remove special tokens before pre-tokenize
    split_pattern = "|".join(map(re.escape, special_tokens))
    chunks = re.split(split_pattern, content)
    # print(chunks)

    # pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # [optimization] build init corpus 
    byte_vocab = defaultdict(int)
    byte_corpus: list[list[str]] = []
    for content in chunks:
        for match in re.finditer(PAT, content):
            # print(f"Found: {match.group()} at {match.start()}–{match.end()}")
            # byte_vocab[tuple(c.encode('utf-8') for c in match.group())] += 1
            byte_vocab[tuple(bytes([i]) for i in match.group().encode('utf-8'))] += 1
            byte_corpus.append(list(bytes([i]) for i in match.group().encode('utf-8')))
    # print(byte_corpus)
    # print(byte_vocab)

    # [optimization] get init stats
    pair_freq, pair_positions = get_initial_stats(byte_corpus)
    # print(pair_freq)
    # print(pair_positions)

    # compute BPE merges
    num_merges = vocab_size - len(vocab)

    # [optimization] new train loop
    for index in range(len(vocab), len(vocab) + num_merges):
        best = max(pair_freq, key=lambda k: (pair_freq[k], k))

        byte_corpus, pair_freq, pair_positions = merge_pair(best, byte_corpus, pair_freq, pair_positions) # inplace edit

        vocab[index] = best[0] + best[1]
        merges.append(best)

    # for index in range(len(vocab), len(vocab) + num_merges):
    #     pairs = get_stats(byte_vocab)
    #     # print(pairs)
    #     best = max(pairs, key=lambda k: (pairs[k], k))
    #     byte_vocab = merge_vocab(best, byte_vocab)
    #     # print(best)
    #     vocab[index] = best[0] + best[1]
    #     merges.append(best)
    # print(vocab)
    # print(merges)
    return vocab, merges

def main():
    input_path = "/mnt/disk3/yusheng/assignment1-basics/data/my_test.txt"
    # input_path = "/mnt/disk3/yusheng/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 256 + 69
    # vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    bpe_train(input_path, vocab_size, special_tokens)

if __name__ == "__main__":
    main()