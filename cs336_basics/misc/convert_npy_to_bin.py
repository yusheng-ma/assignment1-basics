import numpy as np
import os

def convert_npy_to_bin(input_path, output_path, dtype="uint16"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading .npy file: {input_path}")
    data = np.load(input_path)

    print(f"Original dtype: {data.dtype}, shape: {data.shape}")
    data = data.astype(dtype)

    print(f"Saving to .bin file: {output_path} with dtype {dtype}")
    data.tofile(output_path)
    print("Conversion complete.\n")

def main():
    base_dir = "./data/tokenizer"

    file_pairs = [
        ("train_token_ids_10000.npy", "train_token_ids_10000.bin"),
        ("valid_token_ids_10000.npy", "valid_token_ids_10000.bin"),
        ("train_token_ids_32000.npy", "train_token_ids_32000.bin"),
        ("valid_token_ids_32000.npy", "valid_token_ids_32000.bin"),
    ]

    for npy_file, bin_file in file_pairs:
        input_path = os.path.join(base_dir, npy_file)
        output_path = os.path.join(base_dir, bin_file)
        convert_npy_to_bin(input_path, output_path)

if __name__ == "__main__":
    main()
