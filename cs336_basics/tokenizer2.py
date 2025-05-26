import re
from collections import Counter, defaultdict

corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

elements = ['<|endoftext|>'] + [bytes([i]) for i in range(256)]

freq_table = Counter(corpus.split())

byte_vocab = {
    tuple(c.encode('utf-8') for c in word): count
    for word, count in freq_table.items()
}

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

num_merges = 6

for i in range(num_merges):
    pairs = get_stats(byte_vocab)
    best = max(pairs, key=lambda k: (pairs[k], k))
    byte_vocab = merge_vocab(best, byte_vocab)
    print(best)
    elements.append(best[0] + best[1])

print(byte_vocab)
print(elements)