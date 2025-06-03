import wandb
import time

# Initialize a new wandb run
wandb.init(
    project="hello-world",  # Replace with your project name
    name="first-run",       # Optional run name
    config={
        "learning_rate": 0.01,
        "epochs": 5
    }
)

# Access config values
config = wandb.config

# Dummy training loop
for epoch in range(config.epochs):
    loss = 1.0 / (epoch + 1)       # Simulated decreasing loss
    accuracy = epoch / config.epochs  # Simulated increasing accuracy

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })

    time.sleep(1)  # Simulate time-consuming training

# Finish the run
wandb.finish()
