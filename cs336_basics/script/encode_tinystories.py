import numpy as np
import pickle
from tqdm import tqdm
from cs336_basics.tokenizer_class import Tokenizer

if __name__ == "__main__":
    # 直接從 pkl 載入 vocab 和 merges
    vocab_pkl = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl"
    merges_pkl = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl"

    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)

    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges)

    train_filepath = "/mnt/disk3/yusheng/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    valid_filepath = "/mnt/disk3/yusheng/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

    train_output_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/train_token_ids_10000.npy"
    valid_output_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.npy"

    # Tokenize training set with progress bar
    with open(train_filepath, 'r', encoding='utf-8') as f:
        train_ids = []
        for ids in tqdm(tokenizer.encode_iterable(f), desc="Tokenizing train set"):
            if isinstance(ids, int):
                train_ids.append(ids)
            else:
                train_ids.extend(ids)
        train_array = np.array(train_ids, dtype=np.uint16)
        np.save(train_output_path, train_array)
        print(f"Saved train token IDs to {train_output_path} ({len(train_array)} tokens)")

    # Tokenize validation set with progress bar
    with open(valid_filepath, 'r', encoding='utf-8') as f:
        valid_ids = []
        for ids in tqdm(tokenizer.encode_iterable(f), desc="Tokenizing valid set"):
            if isinstance(ids, int):
                valid_ids.append(ids)
            else:
                valid_ids.extend(ids)
        valid_array = np.array(valid_ids, dtype=np.uint16)
        np.save(valid_output_path, valid_array)
        print(f"Saved valid token IDs to {valid_output_path} ({len(valid_array)} tokens)")
