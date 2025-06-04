import os
import glob
import torch
import wandb
import argparse
import numpy as np
from einops import rearrange
from cs336_basics.imports import *


def get_latest_checkpoint(path):
    ckpts = glob.glob(os.path.join(path, "ckpt_*.pt"))
    return max(ckpts, key=os.path.getctime) if ckpts else None


def load_dataset(path, vocab_size):
    assert os.path.exists(path), f"Dataset file not found: {path}"
    
    data = np.memmap(path, dtype=np.uint16, mode='r')
    
    # sanity check
    if np.any(data >= vocab_size):
        raise ValueError(f"Dataset contains values >= vocab_size ({vocab_size})")
    
    print(f"Loaded dataset of shape {data.shape} from {path}")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, required=True, help="Size of the vocabulary")
    parser.add_argument("--context_length", type=int, default=128, help="Context length used for both model and data")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of model embeddings")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of feedforward network")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Theta value for Rotary Positional Encoding")

    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--betas", type=eval, default="(0.9, 0.999)", help="Betas for AdamW (use format '(b1, b2)')")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW optimizer")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="Maximum gradient norm (0 to disable)")

    # Training & Checkpoint
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--out", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--src", type=str, default=None, help="Checkpoint directory to resume training from")

    # Dataset & training
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")

    return parser.parse_args()


def main():
    args = parse_args()

    # load dataset with memory mapping
    train_data = load_dataset(args.dataset, args.vocab_size)

    wandb.init(
        project="transformer-training",
        config=vars(args),
        name=f"run-{wandb.util.generate_id()}"
    )

    model = TransformerLM(
        args.vocab_size,
        args.context_length, # used as max_context_length
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        args.lr,
        args.weight_decay,
        args.betas,
        args.eps
    )

    os.makedirs(args.out, exist_ok=True)

    start_epoch = 0
    if args.src:
        ckpt_path = get_latest_checkpoint(args.src)
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}")
            start_epoch = load_checkpoint(ckpt_path, model, optimizer) + 1

    for epoch in range(start_epoch, args.num_train_epochs):
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # token positions
        batch, sequence_length = x.shape
        token_positions = torch.arange(sequence_length, device=args.device).unsqueeze(0).expand(batch, sequence_length)

        logits = model(x, token_positions)

        loss = cross_entropy(
            rearrange(logits, 'b t v -> (b t) v'),
            rearrange(y, 'b t -> (b t)')
        )

        loss.backward()

        gradient_clipping(model.parameters(), args.max_l2_norm)

        wandb.log({"epoch": epoch, "loss": loss})

        ckpt_path = os.path.join(args.out, f"ckpt_{epoch:04d}.pt")
        if os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} already exists. Skipping.")
            continue
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
