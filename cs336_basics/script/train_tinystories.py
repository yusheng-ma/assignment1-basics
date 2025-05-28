import regex as re
import multiprocessing
import os
import pickle
from collections import defaultdict
from typing import BinaryIO, Tuple
from tqdm import tqdm

# ==== Initial Bigram Stats ====
def get_initial_stats(byte_vocab):
    pair_freq = defaultdict(int)
    for word, freq in byte_vocab.items():
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            pair_freq[pair] += freq
    return pair_freq


# ==== Merge Bigram in Vocabulary ====
def merge_pair(pair, byte_vocab, pair_freq, pair_to_byte_vocab):
    replacement = pair[0] + pair[1]
    words = list(pair_to_byte_vocab[pair])

    for word in words:
        freq = byte_vocab[word]
        del byte_vocab[word]

        # Remove old pairs
        for j in range(len(word) - 1):
            old_pair = (word[j], word[j + 1])
            pair_freq[old_pair] -= freq
            if pair_freq[old_pair] <= 0:
                pair_freq.pop(old_pair, None)
            pair_to_byte_vocab[old_pair].discard(word)
            if not pair_to_byte_vocab[old_pair]:
                del pair_to_byte_vocab[old_pair]

        # Merge word
        w_out = []
        j = 0
        while j < len(word):
            if j < len(word) - 1 and (word[j], word[j + 1]) == pair:
                w_out.append(replacement)
                j += 2
            else:
                w_out.append(word[j])
                j += 1

        new_word = tuple(w_out)
        byte_vocab[new_word] += freq

        # Add new pairs
        for j in range(len(new_word) - 1):
            new_pair = (new_word[j], new_word[j + 1])
            pair_freq[new_pair] += freq
            pair_to_byte_vocab[new_pair].add(new_word)

    return byte_vocab, pair_freq, pair_to_byte_vocab

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args: Tuple[str, int, int, list[str]]) -> Tuple[defaultdict, list]:
    input_path, start, end, special_tokens = args
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    byte_vocab = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start).decode("utf-8", errors="ignore")
        split_pattern = "|".join(map(re.escape, special_tokens))
        chunks = re.split(split_pattern, raw)

        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                byte_seq = [bytes([i]) for i in match.group().encode("utf-8")]
                token = tuple(byte_seq)
                byte_vocab[token] += 1

    return byte_vocab

# ==== BPE Trainer ====
def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # Add special tokens
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token.encode("utf-8")

    # Find chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    # Parallel processing
    args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, args)
    
    # Merge results
    byte_vocab = defaultdict(int)
    for partial_vocab in results:
        for token, freq in partial_vocab.items():
            byte_vocab[token] += freq

    pair_freq = get_initial_stats(byte_vocab)

    pair_to_byte_vocab = defaultdict(set)
    for word in byte_vocab:
        for pair in zip(word, word[1:]):
            pair_to_byte_vocab[pair].add(word)

    # Run merges
    num_merges = vocab_size - len(vocab)
    for i in tqdm(range(num_merges), desc="Merging BPE pairs", unit="merge"):
        if not pair_freq:
            break
        best_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        byte_vocab, pair_freq, pair_to_byte_vocab = merge_pair(
            best_pair, byte_vocab, pair_freq, pair_to_byte_vocab
        )
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

    return vocab, merges


def save_tokenizer_artifacts(vocab, merges, vocab_size, output_dir="data/tokenizer"):
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, f"vocab_{vocab_size}.pkl")
    merges_path = os.path.join(output_dir, f"merges_{vocab_size}.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"✅ Saved vocab to {vocab_path}")
    print(f"✅ Saved merges to {merges_path}")

if __name__ == "__main__":
    input_path = "/mnt/disk3/yusheng/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # input_path = "/mnt/disk3/yusheng/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_processes = 64

    vocab, merges = bpe_train(input_path, vocab_size, special_tokens, num_processes)

    save_tokenizer_artifacts(vocab, merges, vocab_size)
