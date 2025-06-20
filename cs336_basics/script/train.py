import os
import glob
import time
import torch
import wandb
import argparse
import numpy as np
from einops import rearrange
from cs336_basics.imports import *
from muon import SingleDeviceMuonWithAuxAdam

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


@torch.no_grad()
def evaluate_model(model, data, batch_size, context_length, device, num_batches=3):
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        batch, seq_len = x.shape
        token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        logits = model(x, token_positions)
        loss = cross_entropy(
            rearrange(logits, 'b t v -> (b t) v'),
            rearrange(y, 'b t -> (b t)')
        )
        losses.append(loss.item())
        # Explicit cleanup
        # del x, y, logits, loss, token_positions
        # torch.cuda.empty_cache()

    return np.mean(losses)


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

    # Learning rate schedule
    parser.add_argument("--max_learning_rate", type=float, default=5e-4, help="Maximum LR for cosine schedule")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="Minimum LR after cosine annealing")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000, help="Total number of steps for cosine decay")

    # Training & Checkpoint
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--out", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--src", type=str, default=None, help="Checkpoint directory to resume training from")
    parser.add_argument("--keep_last_n_ckpt", type=int, default=5, help="Number of latest checkpoints to keep")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N batchs")

    # Dataset & training
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dataset", type=str, default=None, help="Path to validation dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")

    # Ablation
    parser.add_argument("--no_rmsnorm", action="store_true", help="Disable RMSNorm (use LayerNorm instead)")
    parser.add_argument("--post_norm", action="store_true", help="Use Post-Norm instead of Pre-Norm")
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Encoding")
    parser.add_argument("--act_fn", type=str, default="swiglu", choices=["swiglu", "silu"], help="Activation function for FFN")

    # Muon
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--max_muon_lr", type=float, default=5e-4, help="Maximum LR for cosine schedule")
    parser.add_argument("--min_muon_lr", type=float, default=1e-5, help="Minimum LR after cosine annealing")

    return parser.parse_args()


def main():
    args = parse_args()

    # load dataset with memory mapping
    train_data = load_dataset(args.train_dataset, args.vocab_size)
    val_data = load_dataset(args.val_dataset, args.vocab_size) if args.val_dataset else None

    wandb.init(
        project="transformer-training-owt_muon",
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
        args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        use_rope=not args.no_rope,
        activation=args.act_fn
    ).to(args.device)

    hidden_weights = []
    hidden_gains_biases = []
    nonhidden_params = []

    for name, param in model.named_parameters():
        if 'lm_head' in name or 'token_embeddings' in name:
            nonhidden_params.append(param)
        elif param.ndim >= 2:
            hidden_weights.append(param)
        else:
            hidden_gains_biases.append(param)

    param_groups = [
        dict(params=hidden_weights, use_muon=True,
            lr=args.muon_lr, weight_decay=args.weight_decay), # *10 learning rate, ref: https://kexue.fm/archives/10592
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
            lr=args.lr, betas=args.betas, weight_decay=args.weight_decay),
    ]
    
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    # for group in optimizer.param_groups:
    #     group["initial_lr"] = group["lr"]

    os.makedirs(args.out, exist_ok=True)

    start_time = time.time()
    start_epoch = 0
    step = 0

    if args.src:
        ckpt_path = get_latest_checkpoint(args.src)
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}")
            start_epoch = load_checkpoint(ckpt_path, model, optimizer) + 1
            step = start_epoch

    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        learning_rate = lr_cosine_schedule(
            epoch,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters
        )

        muon_lr = lr_cosine_schedule(
            epoch,
            args.max_muon_lr,
            args.min_muon_lr,
            args.warmup_iters,
            args.cosine_cycle_iters
        )

        for param_group in optimizer.param_groups:
            if param_group.get("use_muon", False):
                param_group["lr"] = muon_lr
            else:
                param_group["lr"] = learning_rate

        # token positions
        batch, sequence_length = x.shape
        token_positions = torch.arange(sequence_length, device=args.device).unsqueeze(0).expand(batch, sequence_length)

        logits = model(x, token_positions)

        loss = cross_entropy(
            rearrange(logits, 'b t v -> (b t) v'),
            rearrange(y, 'b t -> (b t)')
        )

        optimizer.zero_grad()

        loss.backward()

        gradient_clipping(model.parameters(), args.max_l2_norm)

        optimizer.step()

        log_data = {
            "epoch": epoch,
            "step": step,
            "loss": loss.item(),
            "lr": learning_rate,
            "muon_lr": muon_lr,
            "wallclock_time_sec": time.time() - start_time
        }

        # eval val every 5
        if epoch % 5 == 0 and val_data is not None:
            val_loss = evaluate_model(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device,
                num_batches=3
            )
            log_data["val_loss"] = val_loss
            print(f"Epoch {epoch} | train: {loss.item():.4f} | val: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch} | train: {loss.item():.4f}")

        wandb.log(log_data)

        if (epoch + 1) % args.save_every == 0 or epoch == args.num_train_epochs - 1:
            ckpt_path = os.path.join(args.out, f"ckpt_{epoch:04d}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            # save latest n
            ckpts = sorted(
                glob.glob(os.path.join(args.out, "ckpt_*.pt")),
                key=os.path.getmtime
            )
            if len(ckpts) > args.keep_last_n_ckpt:
                for ckpt_to_delete in ckpts[:-args.keep_last_n_ckpt]:
                    try:
                        os.remove(ckpt_to_delete)
                    except Exception as e:
                        print(f"Warning: failed to delete {ckpt_to_delete}: {e}")

        # del x, y, logits, loss, token_positions
        # torch.cuda.empty_cache()

        step += 1

    wandb.finish()

if __name__ == "__main__":
    main()
