import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import keyboard
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from .models.llama import MiniLlamaModel
from .dataset import LlamaDataset
from .tokenizer.llama_tokenizer import LlamaTokenizer

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

def plot_losses(train_losses, val_losses, train_loader_len):
    plt.clf()
    plt.plot(train_losses, label='Train Loss', color='tab:blue')
    if val_losses:
        val_x = [i * train_loader_len for i in range(len(val_losses))]
        plt.plot(val_x, val_losses, label='Validation Loss', color='tab:orange', marker='o')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.tight_layout()
    plt.pause(0.01)

def check_disk_space(path, min_gb=1):
    """Check if there's enough disk space (in GB)"""
    free_space_gb = shutil.disk_usage(path).free / (1024**3)
    if free_space_gb < min_gb:
        print(f"⚠️  WARNING: Only {free_space_gb:.2f}GB free space left!", flush=True)
        return False
    return True

def cleanup_old_checkpoints(checkpoint_dir, keep=10):
    """Keep only the most recent checkpoints"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and 'step' in f]
    if len(checkpoints) > keep:
        checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        to_remove = checkpoints[:-keep]
        for file in to_remove:
            file_path = os.path.join(checkpoint_dir, file)
            os.remove(file_path)
            print(f"🗑️  Removed old checkpoint: {file}", flush=True)

def safe_checkpoint_save(model, checkpoint_path, max_checkpoints=10):
    """Save checkpoint with disk space check and cleanup"""
    try:
        if not check_disk_space(os.path.dirname(checkpoint_path), min_gb=0.5):
            print("🗑️  Cleaning up old checkpoints due to low disk space...", flush=True)
            cleanup_old_checkpoints(os.path.dirname(checkpoint_path), keep=5)
        
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}", flush=True)
        cleanup_old_checkpoints(os.path.dirname(checkpoint_path), keep=max_checkpoints)
        
    except RuntimeError as e:
        if "file write failed" in str(e):
            print(f"💥 DISK FULL! Failed to save checkpoint", flush=True)
            cleanup_old_checkpoints(os.path.dirname(checkpoint_path), keep=3)
        else:
            raise e

def inference_sample(model, tokenizer, device, max_length=50):
    model.eval()
    input_text = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long).to(device)
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=-1)
    print(f"\n[Sample Generation] {tokenizer.decode(generated[0].tolist())}\n", flush=True)

def menu():
    print("\nMini-LLaMA Training Menu")
    print("1. Start new training")
    print("2. Resume training")
    print("3. Train from best model")
    print("4. Fine-tune")
    print("5. Exit")
    return input("Select option (1-5): ")

def get_checkpoint_path(checkpoint_dir, mode):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    if mode == "resume":
        return os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
    elif mode == "best":
        best = sorted(checkpoints, key=lambda x: float(x.split('loss')[-1].rstrip('.pt')))[0]
        return os.path.join(checkpoint_dir, best)
    return None

def main():
    # --- Load configs ---
    with open("configs/model.yaml") as f:
        model_config = yaml.safe_load(f)["model"]
    with open("configs/train.yaml") as f:
        train_config = yaml.safe_load(f)["train"]

    # --- Paths from config ---
    paths = train_config.get("paths", {})
    data_path = paths.get("data", "data/corpus_ids.npy")
    tokenizer_path = paths.get("tokenizer", "tokenizer/llama.model")
    checkpoint_dir = paths.get("checkpoints", "checkpoints")

    # --- Data and split ---
    dataset = LlamaDataset(data_path)
    val_size = int(len(dataset) * train_config["validation_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=train_config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_config["batch_size"], drop_last=False)

    # --- Model, optimizer, etc. ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniLlamaModel(model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"])
    )
    
    # Anti-overfitting features
    criterion = LabelSmoothingLoss(model_config['vocab_size'], smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    tokenizer = LlamaTokenizer(tokenizer_path)

    # --- Dataset info ---
    print(f"\n📊 Dataset Info:")
    print(f"Total sequences: {len(dataset)}")
    print(f"Training sequences: {len(train_ds)}")
    print(f"Validation sequences: {len(val_ds)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # --- Menu ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0
    patience = 5
    patience_counter = 0

    while True:
        choice = menu()
        if choice == "1":
            print("\n[INFO] Starting new training from scratch.", flush=True)
            break
        elif choice == "2":
            path = get_checkpoint_path(checkpoint_dir, "resume")
            if path:
                print(f"[INFO] Resuming from checkpoint: {path}", flush=True)
                model.load_state_dict(torch.load(path))
                break
            else:
                print("[WARN] No checkpoints found.", flush=True)
        elif choice == "3":
            path = get_checkpoint_path(checkpoint_dir, "best")
            if path:
                print(f"[INFO] Loading best model: {path}", flush=True)
                model.load_state_dict(torch.load(path))
                break
            else:
                print("[WARN] No best model found.", flush=True)
        elif choice == "4":
            path = get_checkpoint_path(checkpoint_dir, "best")
            if path:
                print(f"[INFO] Fine-tuning from best model: {path}", flush=True)
                model.load_state_dict(torch.load(path))
                optimizer.param_groups[0]['lr'] *= 0.1  # Fixed bug here
                break
            else:
                print("[WARN] No best model found.", flush=True)
        elif choice == "5":
            print("Exiting.", flush=True)
            return
        else:
            print("Invalid option, try again.", flush=True)

    # --- Training loop ---
    plt.ion()
    train_losses, val_losses = [], []
    step = 0
    start_time = time.time()
    total_batches = train_config["epochs"] * len(train_loader)
    
    inference_interval = int(train_config["inference_interval_steps"])
    checkpoint_interval = int(train_config["checkpoint_interval_steps"])
    
    print("\n💡 Press 'M' at any time during training to manually save the current model!", flush=True)
    print("🚀 Starting training...\n", flush=True)

    for epoch in range(start_epoch, train_config["epochs"]):
        model.train()
        total_train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            
            # Gradient clipping to prevent overfitting
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_losses.append(loss.item())
            step += 1

            # ETA calculation
            elapsed = time.time() - start_time
            batches_done = epoch * len(train_loader) + batch_idx + 1
            eta = format_time((elapsed / batches_done) * (total_batches - batches_done))

            # Real-time plot every 10 batches
            if batch_idx % 10 == 0:
                plot_losses(train_losses, val_losses, len(train_loader))
            
            # Manual save check
            if keyboard.is_pressed('m') or keyboard.is_pressed('M'):
                manual_path = f"{checkpoint_dir}/mini-llama-manual-step{step}.pt"
                safe_checkpoint_save(model, manual_path, max_checkpoints=20)
                print(f"🎯 MANUAL SAVE! Saved at step {step}", flush=True)
            
            # Inference every N steps
            if step % inference_interval == 0:
                inference_sample(model, tokenizer, device)
            
            # Checkpoint every N steps
            if step % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/mini-llama-step{step}.pt"
                safe_checkpoint_save(model, checkpoint_path, max_checkpoints=10)

            # Print log
            print(f"\rEpoch: {epoch+1}/{train_config['epochs']} | Batch: {batch_idx+1}/{len(train_loader)} | "
                  f"Train loss: {loss.item():.4f} | Val loss: " +
                  (f"{val_losses[-1]:.4f}" if val_losses else "N/A") + 
                  f" | ETA: {eta}", end="", flush=True)

        # --- Validation phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits = model(x_val)
                vloss = criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                total_val_loss += vloss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = f"{checkpoint_dir}/mini-llama-best_loss{best_val_loss:.4f}.pt"
            safe_checkpoint_save(model, best_path, max_checkpoints=5)
            patience_counter = 0
        else:
            patience_counter += 1

        # Save epoch checkpoint
        epoch_path = f"{checkpoint_dir}/mini-llama-epoch{epoch+1}_loss{avg_val_loss:.4f}.pt"
        safe_checkpoint_save(model, epoch_path, max_checkpoints=5)

        # Epoch completion log
        print(f"\n✅ Epoch {epoch+1}/{train_config['epochs']} Complete | "
              f"Final Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Best Val: {best_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement.", flush=True)
            print(f"Best validation loss achieved: {best_val_loss:.4f}", flush=True)
            break

    plt.ioff()
    plt.show()

    # Final model info
    print("\n🎉 Training Complete!")
    print(f"Total training time: {format_time(time.time() - start_time)}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("\nModel Parameters:", flush=True)
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name}: {param.size()} ({param_count:,} parameters)", flush=True)
    print(f"Total parameters: {total_params:,}", flush=True)

if __name__ == "__main__":
    main()
