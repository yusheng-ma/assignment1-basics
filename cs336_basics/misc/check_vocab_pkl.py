import pickle

vocab_pkl_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl"
expected_vocab_size = 10000

with open(vocab_pkl_path, "rb") as f:
    vocab = pickle.load(f)

print(f"✅ Loaded vocab from: {vocab_pkl_path}")
print(f"🔢 Vocab size: {len(vocab)}")

key_types = set(type(k) for k in vocab.keys())
value_types = set(type(v) for v in vocab.values())

print(f"🔑 Key types: {key_types}")
print(f"🔤 Value types: {value_types}")

try:
    max_id = max(vocab.keys())
    print(f"📈 Max vocab ID: {max_id}")
except Exception as e:
    print(f"⚠️ Cannot compute max key: {e}")

if expected_vocab_size is not None:
    if len(vocab) != expected_vocab_size:
        print(f"❌ Vocab size ({len(vocab)}) != expected ({expected_vocab_size})")
    else:
        print("✅ Vocab size matches expected")
