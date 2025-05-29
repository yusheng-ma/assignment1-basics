import numpy as np
import pickle
import time
from tqdm import tqdm
from cs336_basics.tokenizer_class import Tokenizer

def compute_compression_ratio(vocab_pkl, merges_pkl, text_file, num_docs=10):
    # Load tokenizer
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges)

    # Read and split text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    documents = [doc for doc in text.split("<|endoftext|>") if doc.strip()]
    print("Full documents count:", len(documents))
    sample_docs = documents[:num_docs]

    total_bytes = 0
    valid_ids = []

    start_time = time.time()

    for doc in tqdm(sample_docs, desc="Tokenizing documents"):
        total_bytes += len(doc.encode('utf-8'))
        ids = tokenizer.encode(doc)
        if isinstance(ids, int):
            valid_ids.append(ids)
        else:
            valid_ids.extend(ids)

    elapsed_time = time.time() - start_time

    total_tokens = len(valid_ids)
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else float('inf')
    throughput = total_bytes / elapsed_time if elapsed_time > 0 else float('inf')

    print(f"\nProcessed {len(sample_docs)} documents")
    print(f"Total bytes: {total_bytes}")
    print(f"Total tokens: {total_tokens}")
    print(f"Compression ratio (bytes/token): {compression_ratio:.4f}")
    print(f"Throughput: {throughput:.2f} bytes/second")

    return compression_ratio, total_bytes, total_tokens, throughput

if __name__ == "__main__":
    # TinyStories Example
    compute_compression_ratio(
        vocab_pkl="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl",
        merges_pkl="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl",
        text_file="/mnt/disk3/yusheng/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
        num_docs=10
    )

    # OpenWebText Example
    compute_compression_ratio(
        vocab_pkl="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_32000.pkl",
        merges_pkl="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_32000.pkl",
        text_file="/mnt/disk3/yusheng/assignment1-basics/data/owt_valid.txt",
        num_docs=10
    )

    # Tokenize OpenWebText with TinyStories tokenizer
    compute_compression_ratio(
        vocab_pkl="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl",
        merges_pkl="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl",
        text_file="/mnt/disk3/yusheng/assignment1-basics/data/owt_valid.txt",
        num_docs=10
    )
