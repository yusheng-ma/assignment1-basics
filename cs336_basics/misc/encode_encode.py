import pickle
from cs336_basics.tokenizer_class import Tokenizer

# ---init tokenizer----------------------
vocab_pkl = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl"
merges_pkl = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl"

with open(vocab_pkl, "rb") as f:
    vocab = pickle.load(f)

with open(merges_pkl, "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
# ---------------------------------------

text_filepath = "/mnt/disk3/yusheng/assignment1-basics/data/my_test.txt"

with open(text_filepath) as f:
    text = f.read()
    ids = tokenizer.encode(text)

    print(ids[:10])
    if 256 in ids:
        print("OMG ITS INSIDE")
    else:
        print("HOLY ITS NOT")

    dec = tokenizer.decode(ids)
    print(dec[:10])

    if dec == text:
        print("same")
    else:
        print("nahh")