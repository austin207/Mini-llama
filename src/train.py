import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import keyboard
import psutil
import signal
import sys
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
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
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

def get_memory_usage():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_max_memory = torch.cuda.max_memory_allocated() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_memory = gpu_max_memory = gpu_total = 0
    
    cpu_memory = psutil.virtual_memory().used / 1e9
    cpu_total = psutil.virtual_memory().total / 1e9
    
    return {
        'gpu_used': gpu_memory,
        'gpu_max': gpu_max_memory,
        'gpu_total': gpu_total,
        'cpu_used': cpu_memory,
        'cpu_total': cpu_total
    }

def plot_losses(train_losses, val_losses, train_loader_len, learning_rates=None):
    """Enhanced plotting with memory and learning rate tracking"""
    
    # Fix: Check if figure exists, if not create it, if yes just select it
    if not plt.fignum_exists(1):
        plt.figure(1, figsize=(15, 5) if learning_rates else (10, 5))
    else:
        plt.figure(1)  # Just select existing figure
    
    plt.clf()  # Clear the figure content
    
    # Create subplots
    if learning_rates:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    else:
        ax1 = plt.subplot(1, 1, 1)
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='tab:blue', alpha=0.7)
    if val_losses:
        val_x = [i * train_loader_len for i in range(len(val_losses))]
        ax1.plot(val_x, val_losses, label='Validation Loss', color='tab:orange', marker='o', markersize=3)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    if learning_rates:
        ax2.plot(learning_rates, label='Learning Rate', color='tab:green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)

def check_disk_space(path, min_gb=1):
    """Check if there's enough disk space (in GB)"""
    try:
        free_space_gb = shutil.disk_usage(path).free / (1024**3)
        if free_space_gb < min_gb:
            print(f"⚠️  WARNING: Only {free_space_gb:.2f}GB free space left!", flush=True)
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}", flush=True)
        return True

def cleanup_old_checkpoints(checkpoint_dir, keep=10):
    """Keep only the most recent checkpoints"""
    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and 'step' in f]
        if len(checkpoints) > keep:
            checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
            to_remove = checkpoints[:-keep]
            for file in to_remove:
                file_path = os.path.join(checkpoint_dir, file)
                os.remove(file_path)
                print(f"🗑️  Removed old checkpoint: {file}", flush=True)
    except Exception as e:
        print(f"⚠️  Error cleaning checkpoints: {e}", flush=True)

def safe_checkpoint_save(model, optimizer, scheduler, scaler, checkpoint_path, 
                        train_losses=None, val_losses=None, global_step=0, 
                        current_epoch=0, best_val_loss=float('inf'), max_checkpoints=10):
    """Enhanced checkpoint saving with full training state"""
    try:
        if not check_disk_space(os.path.dirname(checkpoint_path), min_gb=0.5):
            print("🗑️  Cleaning up old checkpoints due to low disk space...", flush=True)
            cleanup_old_checkpoints(os.path.dirname(checkpoint_path), keep=5)
        
        # Comprehensive checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'global_step': global_step,
            'current_epoch': current_epoch,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses or [],
            'val_losses': val_losses or [],
            'memory_info': get_memory_usage()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}", flush=True)
        cleanup_old_checkpoints(os.path.dirname(checkpoint_path), keep=max_checkpoints)
        
    except RuntimeError as e:
        if "file write failed" in str(e) or "disk" in str(e).lower():
            print(f"💥 DISK FULL! Failed to save checkpoint", flush=True)
            cleanup_old_checkpoints(os.path.dirname(checkpoint_path), keep=3)
        else:
            print(f"❌ Checkpoint save error: {e}", flush=True)
            raise e

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint with full training state"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if provided
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return {
            'global_step': checkpoint.get('global_step', 0),
            'current_epoch': checkpoint.get('current_epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', [])
        }
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}", flush=True)
        return None

def inference_sample(model, tokenizer, device, max_length=50):
    """Enhanced inference sampling with better text generation"""
    model.eval()
    input_text = "Once upon a time"
    
    try:
        input_ids = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long).to(device)
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                with autocast('cuda'):
                    logits = model(generated)
                
                # Apply temperature for better generation
                logits = logits[:, -1, :] / 0.8
                probs = torch.softmax(logits, dim=-1)
                
                # Top-p sampling for better quality
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs > 0.9
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices.gather(-1, next_token)
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop on end token (if implemented)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"\n📝 Sample Generation:")
        print(f"   Input: '{input_text}'")
        print(f"   Output: '{generated_text}'")
        print(f"   Length: {len(generated[0])} tokens\n", flush=True)
        
    except Exception as e:
        print(f"❌ Inference error: {e}", flush=True)
        print(f"📝 Sample Generation: [Error during generation]\n", flush=True)
    
    model.train()

def menu():
    print("\n🔥 Mini-LLaMA Training Menu")
    print("1. Start new training")
    print("2. Resume training")
    print("3. Train from best model")
    print("4. Fine-tune")
    print("5. Load and test model")
    print("6. Exit")
    return input("Select option (1-6): ")

def get_checkpoint_path(checkpoint_dir, mode):
    """Enhanced checkpoint selection"""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
        
    if mode == "resume":
        # Get most recent checkpoint
        checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, checkpoints[-1])
    elif mode == "best":
        # Get best model based on validation loss
        best_checkpoints = [f for f in checkpoints if 'best' in f]
        if best_checkpoints:
            return os.path.join(checkpoint_dir, best_checkpoints[0])
        # Fallback to loss-based selection
        loss_checkpoints = [f for f in checkpoints if 'loss' in f]
        if loss_checkpoints:
            best = min(loss_checkpoints, key=lambda x: float(x.split('loss')[-1].rstrip('.pt')))
            return os.path.join(checkpoint_dir, best)
    return None

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown"""
    print(f"\n⚠️  Received signal {signum}. Attempting graceful shutdown...")
    # Set a global flag that the training loop can check
    global graceful_shutdown
    graceful_shutdown = True

def print_system_info():
    """Print detailed system information"""
    print("🖥️  System Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1e9:.1f}GB")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    
    memory = psutil.virtual_memory()
    print(f"   CPU RAM: {memory.total / 1e9:.1f}GB")
    print(f"   CPU cores: {psutil.cpu_count()}")

def main():
    global graceful_shutdown
    graceful_shutdown = False
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print system info
    print_system_info()
    
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

    # --- Enhanced Data and split ---
    print("📊 Setting up data...")
    
    # Get subset training parameters
    max_sequences = train_config.get("max_sequences", None)
    sequence_length = model_config.get("max_seq_len", 1024)
    
    # Create dataset with enhanced loading
    dataset = LlamaDataset(
        data_path, 
        max_sequences=max_sequences,
        sequence_length=sequence_length
    )
    
    val_size = int(len(dataset) * train_config["validation_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Enhanced data loaders with multi-worker support
    num_workers = min(4, psutil.cpu_count() - 1)
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_config["batch_size"], 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=train_config["batch_size"], 
        drop_last=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=torch.cuda.is_available()
    )

    # --- Enhanced Model, optimizer, etc. ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniLlamaModel(model_config).to(device)
    
    # Model compilation for PyTorch 2.0+
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model)
            print("⚡ Model compiled with PyTorch 2.0")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}")
    
    # Enhanced optimizer with better defaults
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Enhanced learning rate scheduling
    total_steps = train_config["epochs"] * len(train_loader)
    use_onecycle = train_config.get("use_onecycle_lr", False)
    
    if use_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(train_config["lr"]),
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        print("📈 Using OneCycleLR scheduler")
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        print("📈 Using ReduceLROnPlateau scheduler")
    
    # Mixed precision training
    use_amp = train_config.get("use_amp", torch.cuda.is_available())
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("⚡ Mixed precision training enabled")
    
    # Gradient accumulation
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    effective_batch_size = train_config["batch_size"] * gradient_accumulation_steps
    print(f"🎯 Effective batch size: {effective_batch_size}")
    
    # Anti-overfitting features
    criterion = LabelSmoothingLoss(model_config['vocab_size'], smoothing=0.1)
    tokenizer = LlamaTokenizer(tokenizer_path)

    # --- Enhanced Dataset info ---
    print(f"\n📊 Dataset Information:")
    print(f"   Data file: {data_path}")
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024**3)
        print(f"   File size: {file_size:.2f} GB")
    
    if max_sequences:
        print(f"   🎯 Subset training: {max_sequences:,} sequences")
    else:
        print(f"   📈 Full dataset training")
        
    print(f"   Total sequences: {len(dataset):,}")
    print(f"   Training sequences: {len(train_ds):,}")
    print(f"   Validation sequences: {len(val_ds):,}")
    print(f"   Batches per epoch: {len(train_loader):,}")
    print(f"   Validation batches: {len(val_loader):,}")
    print(f"   Device: {device}")
    
    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1e9:.2f}GB (FP32)\n")

    # --- Enhanced Menu ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0
    patience = train_config.get("patience", 5)
    patience_counter = 0
    
    # Training state restoration
    train_losses, val_losses, learning_rates = [], [], []

    while True:
        choice = menu()
        if choice == "1":
            print("\n[INFO] Starting new training from scratch.", flush=True)
            break
        elif choice == "2":
            path = get_checkpoint_path(checkpoint_dir, "resume")
            if path:
                print(f"[INFO] Resuming from checkpoint: {path}", flush=True)
                state = load_checkpoint(path, model, optimizer, scheduler, scaler)
                if state:
                    global_step = state['global_step']
                    start_epoch = state['current_epoch']
                    best_val_loss = state['best_val_loss']
                    train_losses = state['train_losses']
                    val_losses = state['val_losses']
                    print(f"[INFO] Resumed from step {global_step}, epoch {start_epoch}")
                break
            else:
                print("[WARN] No checkpoints found.", flush=True)
        elif choice == "3":
            path = get_checkpoint_path(checkpoint_dir, "best")
            if path:
                print(f"[INFO] Loading best model: {path}", flush=True)
                load_checkpoint(path, model)
                break
            else:
                print("[WARN] No best model found.", flush=True)
        elif choice == "4":
            path = get_checkpoint_path(checkpoint_dir, "best")
            if path:
                print(f"[INFO] Fine-tuning from best model: {path}", flush=True)
                load_checkpoint(path, model, optimizer)
                # Reduce learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"[INFO] Learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}")
                break
            else:
                print("[WARN] No best model found.", flush=True)
        elif choice == "5":
            path = get_checkpoint_path(checkpoint_dir, "best")
            if path:
                print(f"[INFO] Loading model for testing: {path}", flush=True)
                load_checkpoint(path, model)
                inference_sample(model, tokenizer, device, max_length=100)
                continue
            else:
                print("[WARN] No model found.", flush=True)
        elif choice == "6":
            print("Exiting.", flush=True)
            return
        else:
            print("Invalid option, try again.", flush=True)

    # --- Enhanced Training loop ---
    plt.ion()
    step = global_step
    start_time = time.time()
    total_batches = (train_config["epochs"] - start_epoch) * len(train_loader)
    
    inference_interval = int(train_config["inference_interval_steps"])
    checkpoint_interval = int(train_config["checkpoint_interval_steps"])
    
    print("\n💡 Press 'M' at any time during training to manually save the current model!")
    print("💡 Send SIGINT (Ctrl+C) for graceful shutdown with checkpoint saving!")
    print("🚀 Starting training...\n", flush=True)

    try:
        for epoch in range(start_epoch, train_config["epochs"]):
            if graceful_shutdown:
                break
                
            model.train()
            total_train_loss = 0
            epoch_start_time = time.time()
            
            for batch_idx, (x, y) in enumerate(train_loader):
                if graceful_shutdown:
                    break
                    
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                # Mixed precision forward pass
                if use_amp:
                    with autocast('cuda'):
                        logits = model(x)
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        loss = loss / gradient_accumulation_steps
                else:
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    # Step-based scheduler
                    if use_onecycle:
                        scheduler.step()
                        learning_rates.append(optimizer.param_groups[0]['lr'])
                    
                    step += 1
                
                total_train_loss += loss.item() * gradient_accumulation_steps
                train_losses.append(loss.item() * gradient_accumulation_steps)

                # ETA calculation
                elapsed = time.time() - start_time
                if step > global_step:
                    steps_done = step - global_step
                    eta = format_time((elapsed / steps_done) * (total_batches - steps_done))
                else:
                    eta = "Calculating..."

                # Memory monitoring
                memory_info = get_memory_usage()

                # Real-time plot every 10 batches
                if batch_idx % 10 == 0:
                    plot_losses(train_losses, val_losses, len(train_loader), learning_rates if use_onecycle else None)
                
                # Manual save check (Linux flag-based)
                flag_path = os.path.join(checkpoint_dir, "manual_save.flag")
                if os.path.exists(flag_path):
                    manual_path = f"{checkpoint_dir}/mini-llama-manual-step{step}.pt"
                    safe_checkpoint_save(
                        model, optimizer, scheduler, scaler, manual_path,
                        train_losses, val_losses, step, epoch, best_val_loss, max_checkpoints=20
                    )
                    os.remove(flag_path)
                
                # Manual save check (Windows keyboard-based) - Uncomment for Windows
                # try:
                #     if keyboard.is_pressed('m') or keyboard.is_pressed('M'):
                #         manual_path = f"{checkpoint_dir}/mini-llama-manual-step{step}.pt"
                #         safe_checkpoint_save(
                #             model, optimizer, scheduler, scaler, manual_path,
                #             train_losses, val_losses, step, epoch, best_val_loss, max_checkpoints=20
                #         )
                #         print(f"🎯 MANUAL SAVE! Saved at step {step}", flush=True)
                # except:
                #     pass
                
                # Inference every N steps
                if step % inference_interval == 0 and step > 0:
                    inference_sample(model, tokenizer, device)
                
                # Checkpoint every N steps
                if step % checkpoint_interval == 0 and step > 0:
                    checkpoint_path = f"{checkpoint_dir}/mini-llama-step{step}.pt"
                    safe_checkpoint_save(
                        model, optimizer, scheduler, scaler, checkpoint_path,
                        train_losses, val_losses, step, epoch, best_val_loss, max_checkpoints=10
                    )

                # Enhanced progress logging
                if batch_idx % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"\rEpoch: {epoch+1}/{train_config['epochs']} | "
                          f"Batch: {batch_idx+1}/{len(train_loader)} | "
                          f"Step: {step} | "
                          f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"GPU: {memory_info['gpu_used']:.1f}GB/{memory_info['gpu_total']:.1f}GB | "
                          f"ETA: {eta}", end="", flush=True)

            if graceful_shutdown:
                break

            # --- Enhanced Validation phase ---
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                    
                    if use_amp:
                        with autocast('cuda'):
                            logits = model(x_val)
                            vloss = criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                    else:
                        logits = model(x_val)
                        vloss = criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                    
                    total_val_loss += vloss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Learning rate scheduling (epoch-based)
            if not use_onecycle:
                scheduler.step(avg_val_loss)

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = f"{checkpoint_dir}/mini-llama-best_loss{best_val_loss:.4f}.pt"
                safe_checkpoint_save(
                    model, optimizer, scheduler, scaler, best_path,
                    train_losses, val_losses, step, epoch, best_val_loss, max_checkpoints=5
                )
                patience_counter = 0
                print("💎 New best model saved!")
            else:
                patience_counter += 1

            # Save epoch checkpoint
            epoch_path = f"{checkpoint_dir}/mini-llama-epoch{epoch+1}_loss{avg_val_loss:.4f}.pt"
            safe_checkpoint_save(
                model, optimizer, scheduler, scaler, epoch_path,
                train_losses, val_losses, step, epoch, best_val_loss, max_checkpoints=5
            )

            # Enhanced epoch completion log
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\n✅ Epoch {epoch+1}/{train_config['epochs']} Complete")
            print(f"   Train Loss: {train_losses[-1]:.4f}")
            print(f"   Val Loss: {avg_val_loss:.4f}")
            print(f"   Best Val: {best_val_loss:.4f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Epoch Time: {format_time(epoch_time)}")
            print(f"   Total Time: {format_time(total_time)}")
            print(f"   Memory Peak: {memory_info['gpu_max']:.1f}GB")
            print(f"   Patience: {patience_counter}/{patience}", flush=True)

            # Early stopping check
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement.")
                print(f"Best validation loss achieved: {best_val_loss:.4f}", flush=True)
                break

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        graceful_shutdown = True
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final checkpoint save
        if not graceful_shutdown:
            final_path = f"{checkpoint_dir}/mini-llama-final-step{step}.pt"
            safe_checkpoint_save(
                model, optimizer, scheduler, scaler, final_path,
                train_losses, val_losses, step, epoch, best_val_loss, max_checkpoints=5
            )

    plt.ioff()
    plt.show()

    memory_info = get_memory_usage()

    # Enhanced final model info
    print("\n🎉 Training Complete!")
    total_training_time = time.time() - start_time
    print(f"📊 Training Summary:")
    print(f"   Total training time: {format_time(total_training_time)}")
    print(f"   Total steps: {step:,}")
    print(f"   Final epoch: {epoch + 1}")
    print(f"   Final train loss: {train_losses[-1]:.4f}" if train_losses else "   No training data")
    print(f"   Final validation loss: {val_losses[-1]:.4f}" if val_losses else "   No validation data")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Peak GPU memory: {memory_info['gpu_max']:.1f}GB")
    
    print("\n🧠 Model Architecture Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"   {name}: {param.size()} ({param_count:,} parameters)")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1e9:.2f}GB (FP32)")

if __name__ == "__main__":
    main()
