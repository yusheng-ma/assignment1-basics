import numpy as np
import os

# ====== 你可以在這裡修改設定 ======
input_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/train_token_ids_10000.bin"
output_path = "/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/small_train_1024.bin"
num_tokens = 1024
# =================================

print(f"Loading from {input_path}...")
data = np.fromfile(input_path, dtype=np.uint16)
print(f"Original token count: {len(data)}")

subset = data[:num_tokens]
print(f"Extracted {len(subset)} tokens.")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
subset.tofile(output_path)
print(f"Saved to {output_path}")
