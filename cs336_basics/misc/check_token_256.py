import numpy as np
from cs336_basics.tokenizer_class import Tokenizer

# 設定檔案路徑與目標 ID
BIN_PATH = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.bin"
NPY_PATH = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.npy"
TARGET_ID = 256

# 檢查 bin 檔案
token_ids_bin = np.fromfile(BIN_PATH, dtype=np.uint16)
print(f"[BIN] 前 100 個 token: {token_ids_bin[:100]}")
if TARGET_ID in token_ids_bin:
    print(f"✅ [BIN] Token ID {TARGET_ID} 存在於資料中。")
else:
    print(f"❌ [BIN] Token ID {TARGET_ID} 不存在於資料中。")

# 檢查 npy 檔案
token_ids_npy = np.load(NPY_PATH)
print(f"[NPY] 前 100 個 token: {token_ids_npy[:100]}")
if TARGET_ID in token_ids_npy:
    print(f"✅ [NPY] Token ID {TARGET_ID} 存在於資料中。")
else:
    print(f"❌ [NPY] Token ID {TARGET_ID} 不存在於資料中。")
