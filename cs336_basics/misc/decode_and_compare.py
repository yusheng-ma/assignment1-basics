import numpy as np
import pickle
from cs336_basics.tokenizer_class import Tokenizer

if __name__ == "__main__":
    # 讀取 vocab 和 merges
    vocab_pkl = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl"
    merges_pkl = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl"

    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)

    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges)

    # 載入 token ID 的 .npy 檔
    input_token_ids_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.npy"
    output_text_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/decoded_valid_text.txt"

    token_ids = np.load(input_token_ids_path)

    # 還原成文字
    decoded_text = tokenizer.decode(token_ids.tolist())

    # 儲存為 .txt
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(decoded_text)

    print(f"✅ 已儲存還原文字到: {output_text_path}")

    # 原始文字檔案路徑
    origin_text_path = "/mnt/disk3/yusheng/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

    # 比較整體內容是否一致
    print("\n🔍 正在比對整體內容是否一致...")
    with open(origin_text_path, "r", encoding="utf-8") as f1, open(output_text_path, "r", encoding="utf-8") as f2:
        origin_all = f1.read().strip()
        decoded_all = f2.read().strip()

        if origin_all == decoded_all:
            print("🎉 兩個檔案內容完全一致！")
        else:
            print("❌ 兩個檔案內容不一致。")
