import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bi_gram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bi_gram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out

vocab = {
    'l o w': 5,
    'l o w e r': 2,
    'n e w e s t': 6,
    'w i d e s t': 3,
}

num_merges = 10

print(vocab)
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

print(vocab)