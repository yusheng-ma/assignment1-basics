import regex as re
from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for byte_word, freq in vocab.items():
        for pair in zip(byte_word, byte_word[1:]):
            pairs[pair] += freq
    return pairs

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
    
    byte_vocab = defaultdict(int)
    for content in chunks:
        for match in re.finditer(PAT, content):
            # print(f"Found: {match.group()} at {match.start()}–{match.end()}")
            # byte_vocab[tuple(c.encode('utf-8') for c in match.group())] += 1
            byte_vocab[tuple(bytes([i]) for i in match.group().encode('utf-8'))] += 1
    # print(byte_vocab)

    # compute BPE merges
    num_merges = vocab_size - len(vocab)

    for index in range(len(vocab), len(vocab) + num_merges):
        pairs = get_stats(byte_vocab)
        # print(pairs)
        best = max(pairs, key=lambda k: (pairs[k], k))
        byte_vocab = merge_vocab(best, byte_vocab)
        # print(best)
        vocab[index] = best[0] + best[1]
        merges.append(best)
    # print(vocab)

    return vocab, merges

def main():
    # input_path = "/mnt/disk3/yusheng/assignment1-basics/data/my_test.txt"
    input_path = "/mnt/disk3/yusheng/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    # vocab_size = 256 + 69
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    bpe_train(input_path, vocab_size, special_tokens)

if __name__ == "__main__":
    main()