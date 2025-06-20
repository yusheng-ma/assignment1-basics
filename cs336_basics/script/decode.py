import torch
import argparse
import pickle
from cs336_basics.tokenizer_class import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.softmax import softmax


def top_p_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Include tokens until cumulative sum >= p
    cutoff = torch.searchsorted(cumulative_probs, p).item() + 1

    top_p_probs = sorted_probs[:cutoff]
    top_p_indices = sorted_indices[:cutoff]

    top_p_probs = top_p_probs / top_p_probs.sum()
    sampled = torch.multinomial(top_p_probs, 1)
    return top_p_indices[sampled]


@torch.no_grad()
def generate(
    prompt,
    model,
    tokenizer,
    eos_token_id,
    max_context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.0,
    device='cuda'
):
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    num_generated = 0
    while True:
        if input_tensor.shape[1] >= max_context_length:
            print(f"⚠️ 已達最大 context 長度 ({max_context_length})，停止生成")
            break
        if num_generated >= max_new_tokens:
            print(f"ℹ️ 已生成最大 token 數 ({max_new_tokens})，停止生成")
            break

        token_positions = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)
        logits = model(input_tensor, token_positions)
        next_token_logits = logits[:, -1, :]

        adjusted_logits = next_token_logits / temperature
        probs = softmax(adjusted_logits, dim=-1)

        if top_p > 0.0:
            next_token = top_p_sample(probs[0], top_p).unsqueeze(0)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        num_generated += 1

        if next_token.item() == eos_token_id:
            break

    return tokenizer.decode(input_tensor[0].tolist())

def load_model(ckpt_path, device, args):
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate from")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    
    parser.add_argument("--vocab_pkl", required=True, help="Path to vocab.pkl file")
    parser.add_argument("--merges_pkl", required=True, help="Path to merges.pkl file")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (lower = more greedy)")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling threshold (e.g. 0.9; 0 = disable)")

    return parser.parse_args()

def main():
    args = parse_args()

    # 讀取 vocab 和 merges（pickle 格式）
    with open(args.vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges_pkl, "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    eos_token_id = tokenizer.vocab_to_int.get(b"<|endoftext|>")
    if eos_token_id is None:
        raise ValueError("<|endoftext|> not found in tokenizer vocab")

    model = load_model(args.ckpt, args.device, args)

    output = generate(
        args.prompt,
        model,
        tokenizer,
        eos_token_id,
        args.context_length,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.device
    )

    print(f"\n=== Prompt ===\n{args.prompt}\n\n=== Completion ===\n{output}")

if __name__ == "__main__":
    main()
