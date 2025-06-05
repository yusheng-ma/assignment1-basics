import json
import regex as re
from typing import Iterable, Optional
from tqdm import tqdm

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = DEFAULT_SPECIAL_TOKENS):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is DEFAULT_SPECIAL_TOKENS:
            special_tokens = list(DEFAULT_SPECIAL_TOKENS)
        self.special_tokens = special_tokens if special_tokens is not None else []

        # add speical_tokens to vocab if not exist
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in self.vocab.values():
                self.vocab[len(self.vocab)] = encoded
        
        # create reverse vocab
        self.vocab_to_int = {v: k for k, v in self.vocab.items()}

        # [optimization]
        self.merges_dict = {merge: order for order, merge in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            # vocab = {int(k): v.encode("utf-8") for k, v in json.load(f).items()}
            vocab = {int(v): k.encode("utf-8") for k, v in json.load(f).items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = []
            for line in f:
                merges.append(tuple(token.encode("utf-8") for token in line.strip().split()))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pretoken
        if self.special_tokens:
            split_pattern = "|".join(
                f"({re.escape(token)})" # reverse token with ()
                for token in sorted(self.special_tokens, key=len, reverse=True)
            ) # stupidily need to match longer token first
            chunks = re.split(split_pattern, text)
            chunks = [chunk for chunk in chunks if chunk is not None] # clean up None (getting None from ()|() operation)
        else:
            chunks = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        int_seq = []
        for chunk in chunks:
            # if special_tokens
            if chunk in self.special_tokens:
                int_seq.append(self.vocab_to_int[chunk.encode("utf-8")])
                continue

            for match in re.finditer(PAT, chunk):
                byte_seq = [bytes([i]) for i in match.group().encode("utf-8")]
                # merge
                # [optimization] no greedy left to right match, think about "hello", we would wrongly have "he" at first merge but instead correct "ll"
                while len(byte_seq) >= 2:
                    candidate_pairs = [(byte1, byte2) for byte1, byte2 in zip(byte_seq, byte_seq[1:]) if (byte1, byte2) in self.merges_dict]
                    best_pair = min(candidate_pairs, key=lambda pair: self.merges_dict[pair]) if candidate_pairs else None

                    if best_pair:
                        # merge best_pair in byte_seq
                        seq_out = []
                        i = 0
                        while i < len(byte_seq):
                            if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i + 1]) == best_pair:
                                seq_out.append(best_pair[0] + best_pair[1])
                                i += 2
                            else:
                                seq_out.append(byte_seq[i])
                                i += 1
                        byte_seq = seq_out
                    else:
                        break

                for vocab in byte_seq:
                    int_seq.append(self.vocab_to_int[vocab])
    
        return int_seq

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in tqdm(iterable, desc="Encoding lines", total=36990): #36990 for mem test
            for id in self.encode(text):
                yield id
 
    def decode(self, ids: list[int]) -> str:
        byte_seq = []
        for id in ids:
            byte_seq.append(self.vocab[id])
        return b"".join(byte_seq).decode("utf-8", errors="replace")

if __name__ == "__main__":
    vocab_filepath = "/mnt/disk3/yusheng/assignment1-basics/tests/fixtures/gpt2_vocab.json"
    merges_filepath = "/mnt/disk3/yusheng/assignment1-basics/tests/fixtures/gpt2_merges.txt"
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    # special_tokens = None
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # test = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    test = "Hello"
    enc = tokenizer.encode(test)
    print(enc)

    dec = tokenizer.decode(enc)
    print(dec)