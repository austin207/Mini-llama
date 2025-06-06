# Mini LLaMA

A minimal yet powerful LLaMA-style transformer implementation optimized for RTX 4000 Ada GPU training with massive datasets and professional inference capabilities.

## 🖥️ Hardware Specifications
- **GPU**: NVIDIA RTX 4000 Ada Generation (20GB VRAM)
- **RAM**: 32GB + 8GB swap (upgraded from 16GB)
- **CPU**: 24 cores
- **Optimized for**: Professional AI/ML training and large-scale language model development

## ⚡ Quick Setup

### **1. Clone and Environment Setup**
```
git clone https://github.com/austin207/Mini-llama.git
cd Mini-llama
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# .\myenv\Scripts\Activate.ps1  # Windows
```

### **2. Install Dependencies**
```
pip install -r requirements.txt
# or for Python 3:
pip3 install -r requirements.txt
```

### **3. Dataset Download Options**
```
# Method 1: Git LFS (Recommended for large datasets)
git lfs install
git clone https://huggingface.co/datasets/faizack/wikipedia-data

# Method 2: Hugging Face CLI
pip install huggingface_hub
huggingface-cli download faizack/wikipedia-data wikipedia_data.txt --repo-type dataset

# Method 3: Python programmatic download
python -c "
from huggingface_hub import hf_hub_download
file_path = hf_hub_download(
    repo_id='faizack/wikipedia-data',
    filename='wikipedia_data.txt',
    repo_type='dataset'
)
print(f'Downloaded to: {file_path}')
"
```

## 📊 Data Preprocessing Pipeline

### **Step 1: Tokenize Raw Text**
```
# Train custom tokenizer (if needed)
python3 -m src.tokenizer.train_tokenizer

# Convert tokenizer format
python3 -m src.tokenizer.converter

# Encode entire corpus with multiprocessing
python3 -m src.tokenizer.encode_corpus
```

### **Step 2: Stream Concatenate Checkpoints**
After preprocessing, you'll have 100+ checkpoint files. Combine them efficiently:
```
cd data/checkpoints
python3 stream_concatenate.py
```
**Result**: Single `corpus_ids_final.dat` file (~41GB, 10.8M sequences)

## ⚙️ Configuration Files

### **configs/model.yaml** - RTX 4000 Ada Optimized
```
model:
  vocab_size: 32000        # SentencePiece standard vocab
  n_layers: 12             # Optimal for 20GB VRAM
  n_heads: 12              # Increased attention capacity
  d_model: 768             # Professional transformer size
  d_ff: 3072               # 4x d_model feedforward
  max_seq_len: 1024        # Matches preprocessing
  dropout: 0.1             # Optimal regularization
```

### **configs/tokenizer.yaml** - SentencePiece Setup
```
tokenizer:
  vocab_size: 32000
  model_type: sentencepiece
  special_tokens: ["[BOS]", "[EOS]", "[PAD]"]
```

### **configs/train.yaml** - Production Training Setup
```
train:
  # Hardware-optimized batch settings
  batch_size: 32                    # Utilizes 20GB VRAM efficiently
  gradient_accumulation_steps: 4    # Effective batch size: 128
  epochs: 30
  lr: 3e-4                         # Optimal for large models
  weight_decay: 0.1
  seed: 42
  
  # Advanced training features
  use_amp: true                    # Mixed precision for RTX 4000 Ada
  use_onecycle_lr: true           # Faster convergence
  patience: 5                     # Early stopping
  
  # Checkpointing and monitoring
  checkpoint_interval_steps: 1000
  inference_interval_steps: 2000
  validation_split: 0.01          # 1% validation (108K sequences)
  
  # Subset training for progressive scaling
  max_sequences: 100000           # Start with 100K, set to null for full 10.8M
  
  paths:
    data: "data/corpus_ids_final.dat"
    tokenizer: "src/tokenizer/llama.model"
    checkpoints: "checkpoints"
```

### **configs/inference.yaml** - Inference Configuration
```
# Inference Configuration for Mini-LLaMA
model:
  # Path to trained model checkpoint
  checkpoint_path: "checkpoints/mini-llama-best_loss1.2347.pt"
  
  # Model configuration
  config_path: "configs/model.yaml"
  
  # Tokenizer configuration  
  tokenizer_path: "src/tokenizer/llama.model"
  
  # Device configuration
  device: "auto"  # "cuda", "cpu", or "auto" for automatic detection

# Default generation parameters
generation:
  max_length: 100
  temperature: 0.8
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.1
  do_sample: true
  num_return_sequences: 1

# Benchmark configuration
benchmark:
  runs_per_prompt: 3
  default_prompts:
    - "Once upon a time"
    - "The future of artificial intelligence is"
    - "In a world where technology"
    - "The quick brown fox"
    - "Explain the concept of machine learning"
    - "Write a short story about"
    - "The most important thing in life is"
    - "Scientists recently discovered"

# Performance monitoring
monitoring:
  enable_timing: true
  show_memory_usage: true
  save_stats: true
  stats_file: "logs/inference_stats.json"
```

## 🚀 Training Execution

### **Progressive Training Strategy**
```
# Phase 1: Quick test (30 minutes)
# Edit train.yaml: max_sequences: 10000
python -m src.train

# Phase 2: Small training (2-4 hours)  
# Edit train.yaml: max_sequences: 100000
python -m src.train

# Phase 3: Medium training (8-12 hours)
# Edit train.yaml: max_sequences: 500000  
python -m src.train

# Phase 4: Full training (2-7 days)
# Edit train.yaml: max_sequences: null
python -m src.train
```

### **Training Menu Options**
The training script provides an interactive menu:
1. **Start new training** - Fresh model training
2. **Resume training** - Continue from last checkpoint  
3. **Train from best model** - Continue from best validation loss
4. **Fine-tune** - Lower learning rate from best model
5. **Load and test model** - Text generation testing
6. **Exit** - Quit training

## 🎯 Inference & Text Generation

### **Quick Start Inference**
```
# Simple text generation
python inference.py --prompt "Once upon a time" --temperature 0.8

# Interactive chat mode
python inference.py --interactive

# Benchmark performance testing
python inference.py --benchmark --runs 5
```

### **Advanced Inference Parameters**

#### **Temperature (0.1-2.0)**
Controls randomness of generation:
- `0.1-0.5`: Very focused, deterministic text
- `0.8`: Balanced creativity (recommended)
- `1.0`: Neutral sampling
- `1.5-2.0`: Very creative, random text

#### **Top-k (0-100)**
Limits token selection to top-k most probable:
- `10-20`: Very focused vocabulary
- `50`: Balanced selection (recommended)
- `0`: Disabled (uses all vocabulary)

#### **Top-p (0.1-1.0)**
Nucleus sampling - cumulative probability threshold:
- `0.8`: Focused selection
- `0.9`: Balanced (recommended)
- `1.0`: Disabled

#### **Repetition Penalty (1.0-2.0)**
Reduces repeated phrases:
- `1.0`: No penalty
- `1.1`: Light penalty (recommended)
- `1.5+`: Strong penalty (may reduce coherence)

### **Inference Usage Examples**

#### **Config-Based Usage (Recommended)**
```
# Use default inference.yaml config
python inference.py --prompt "Explain quantum computing"

# Use custom config file
python inference.py --inference_config configs/my_inference.yaml --interactive

# Override specific parameters
python inference.py --temperature 1.2 --top_k 30 --benchmark
```

#### **Command Line Parameter Override**
```
# Creative generation
python inference.py --prompt "Write a story about AI" \
  --temperature 1.2 --top_k 40 --top_p 0.8 --max_length 200

# Focused technical explanation
python inference.py --prompt "Explain machine learning:" \
  --temperature 0.3 --top_k 10 --repetition_penalty 1.2 --max_length 150

# Multiple creative variations
python inference.py --prompt "The future of technology" \
  --temperature 0.9 --runs 3
```

#### **Interactive Chat Mode**
```
# Start interactive chat with performance monitoring
python inference.py --interactive

# Available chat commands:
# /temp 0.8       - Set temperature
# /topk 40        - Set top-k  
# /topp 0.8       - Set top-p
# /penalty 1.2    - Set repetition penalty
# /length 200     - Set max length
# /params         - Show current settings
# /stats          - Show performance statistics
# /config         - Show configuration info
# /reset          - Reset performance counters
# /quit           - Exit chat
```

#### **Continuous Benchmark Mode**
```
# Run comprehensive performance benchmark
python inference.py --benchmark --runs 5 --max_length 150

# Custom benchmark with specific parameters
python inference.py --benchmark \
  --temperature 0.7 --top_k 40 --runs 3 \
  --inference_config configs/custom_inference.yaml
```

### **Performance Monitoring Features**

The inference script provides comprehensive performance monitoring:

**Real-time Metrics:**
- **Prompt Processing Speed**: Tokens/sec for input processing
- **Generation Speed**: Tokens/sec for output generation  
- **Total Speed**: Overall tokens/sec including both phases
- **Memory Usage**: GPU memory consumption tracking

**Benchmark Statistics:**
- **Average Performance**: Mean tokens/sec across all runs
- **Performance Range**: Min/max generation speeds
- **Timing Breakdown**: Separate prompt vs generation timing
- **Token Statistics**: Average token counts per phase

**Example Output:**
```
🚀 Generated 45 tokens in 0.89s
📊 Performance:
   Prompt Processing: 2,847.3 tokens/sec (0.007s)
   Generation: 50.6 tokens/sec (0.89s)
   Total: 56.2 tokens/sec (0.90s)
   GPU Memory: 8.2GB

📊 Running Average (15 samples):
   Generation: 48.3 tokens/sec
   Range: 42.1 - 53.7 tokens/sec
```

## 💾 Memory-Mapped Dataset Loader

The dataset loader in `src/dataset.py` supports:
- **Memory-mapped .dat files** for 41GB+ datasets
- **Legacy .npy file** support
- **Subset training** for gradual scaling
- **Automatic format detection**

```
# Efficient loading for massive datasets
dataset = LlamaDataset(
    "data/corpus_ids_final.dat", 
    max_sequences=100000,  # Optional subset
    sequence_length=1024
)
```

## 🔥 Advanced Training Features

### **RTX 4000 Ada Optimizations**
- ✅ **Mixed Precision (FP16)**: 2x speedup + memory savings
- ✅ **Gradient Checkpointing**: Fits larger models in 20GB VRAM  
- ✅ **PyTorch 2.0 Compilation**: Automatic optimization
- ✅ **Multi-worker Data Loading**: Efficient CPU utilization

### **Training Quality Features**  
- ✅ **Label Smoothing Loss**: Better generalization
- ✅ **OneCycleLR Scheduler**: Faster convergence
- ✅ **Gradient Clipping**: Training stability
- ✅ **Early Stopping**: Prevent overfitting

### **Monitoring & Management**
- ✅ **Real-time Loss Plotting**: Training/validation curves
- ✅ **Memory Usage Tracking**: GPU/CPU monitoring  
- ✅ **Graceful Shutdown**: Signal handling with checkpoint save
- ✅ **Manual Checkpoint**: Save anytime during training
- ✅ **Automatic Cleanup**: Manage disk space efficiently

## 📈 Expected Performance

### **Model Specifications**
- **Parameters**: ~95M (optimal for 20GB VRAM)
- **Training Speed**: 800-1200 tokens/second
- **Generation Speed**: 40-60 tokens/second
- **Memory Usage**: 14-16GB VRAM during training
- **Context Length**: 1024 tokens

### **Training Timeline**
- **Quick Test**: 30 minutes (10K sequences)
- **Initial Results**: 2-4 hours (100K sequences)  
- **Good Model**: 8-12 hours (500K sequences)
- **Production Model**: 2-7 days (10.8M sequences)

### **Inference Performance**
- **RTX 4000 Ada**: 40-60 tokens/sec generation
- **CPU Fallback**: 5-15 tokens/sec generation
- **Memory Efficient**: ~8GB VRAM for inference
- **Real-time Chat**: Sub-second response times

## 🛠️ Streaming Concatenation Script

Located in `data/stream_concatenate.py`:
- **Memory-efficient**: Processes 41GB without loading into RAM
- **Automatic key detection**: Works with your preprocessing format
- **Progress monitoring**: Real-time statistics
- **Error handling**: Robust file processing

## 📝 Dataset Statistics

Your preprocessed corpus:
- **Total Sequences**: 10,821,218
- **Total Tokens**: ~11 billion  
- **File Size**: 41.28 GB
- **Sequence Length**: 1024 tokens each
- **Format**: Memory-mapped binary (.dat)

## 🔧 Manual Checkpoint Saving

During training, save checkpoints manually:
```
# Linux method: Create flag file
touch checkpoints/manual_save.flag

# The training script will detect and save immediately
```

## 🎯 Hardware Memory Optimization

### **Memory Usage Breakdown**
- **Model Weights**: ~6GB (FP16)
- **Gradients**: ~6GB
- **Activations**: ~6GB (batch_size=32)
- **Buffer**: ~2GB
- **Total**: ~20GB VRAM ✅

### **CPU Memory Management**  
- **Dataset**: Memory-mapped (minimal RAM usage)
- **Preprocessing**: Streaming (no 41GB loading)
- **Checkpoints**: Compressed storage

## 🎮 Usage Workflows

### **Training Workflow**
```
# 1. Preprocess data
python3 -m src.tokenizer.encode_corpus
cd data/checkpoints && python3 stream_concatenate.py

# 2. Start training
cd ../..
python -m src.train
# Select option 1 for new training

# 3. Monitor progress and adjust parameters
# Training will show real-time loss curves and performance metrics
```

### **Inference Workflow**
```
# 1. Quick generation test
python inference.py --prompt "Hello, AI assistant" --temperature 0.8

# 2. Interactive development
python inference.py --interactive
# Use /params, /stats commands to monitor performance

# 3. Performance benchmarking
python inference.py --benchmark --runs 5
# Comprehensive performance analysis across multiple prompts
```

### **Experimentation Workflow**
```
# 1. Start with subset training
# Edit configs/train.yaml: max_sequences: 10000
python -m src.train

# 2. Test inference on quick model
python inference.py --interactive

# 3. Scale up gradually
# Edit configs/train.yaml: max_sequences: 100000
python -m src.train
# Select option 2 to resume from previous checkpoint

# 4. Production deployment
# Edit configs/train.yaml: max_sequences: null
python -m src.train
```

## 📚 Project Structure

```
Mini-llama/
├── configs/
│   ├── model.yaml              # Model architecture
│   ├── train.yaml              # Training configuration
│   ├── tokenizer.yaml          # Tokenizer settings
│   └── inference.yaml          # Inference configuration
├── src/
│   ├── models/
│   │   └── llama.py           # Transformer model implementation
│   ├── tokenizer/
│   │   ├── llama.model        # Trained SentencePiece tokenizer
│   │   ├── train_tokenizer.py # Tokenizer training
│   │   ├── converter.py       # Tokenizer format conversion
│   │   └── encode_corpus.py   # Corpus preprocessing
│   ├── dataset.py             # Memory-mapped dataset loader
│   └── train.py               # Training script
├── data/
│   ├── checkpoints/           # Preprocessing checkpoints
│   └── corpus_ids_final.dat   # Final concatenated corpus
├── checkpoints/               # Model checkpoints
├── logs/                      # Training and inference logs
├── inference.py               # Advanced inference script
├── stream_concatenate.py      # Checkpoint concatenation
└── requirements.txt           # Dependencies
```

## 📚 References & Resources

- **Git LFS**: https://git-lfs.github.com/
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets/
- **PyTorch Mixed Precision**: https://pytorch.org/docs/stable/amp.html
- **Transformer Architecture**: https://arxiv.org/abs/1706.03762
- **SentencePiece**: https://github.com/google/sentencepiece

## 🎉 Success Metrics

Your setup achieves:
- **10-20x faster training** vs CPU-only setups
- **Professional-scale models** (~95M parameters)
- **Enterprise-grade dataset** (10.8M sequences, 11B tokens)
- **Production-ready pipeline** with all optimizations
- **Real-time inference** with advanced sampling techniques
- **Comprehensive monitoring** and performance analytics

## 🔧 Troubleshooting

### **Common Training Issues**
```
# CUDA out of memory
# Reduce batch_size in configs/train.yaml from 32 to 16

# Plot window not showing during training
# Edit src/train.py: add matplotlib backend forcing

# Training interrupted
# Use option 2 "Resume training" to continue from checkpoint
```

### **Common Inference Issues**
```
# Model not found
# Check checkpoint path in configs/inference.yaml

# Slow generation speed
# Ensure CUDA is available: python -c "import torch; print(torch.cuda.is_available())"

# Config file not found
# Create configs/inference.yaml or use command line parameters
```

### **Memory Optimization Tips**
```
# Training: Use gradient accumulation instead of larger batch sizes
# Inference: Use smaller max_length values for longer conversations
# System: Monitor GPU memory with nvidia-smi
```

---

**This README consolidates your entire AI/ML journey from beginner to professional-grade language model training and inference. Everything needed to reproduce, continue, and deploy your work is documented here.** 🚀

*Last Updated: Based on complete implementation including advanced training pipeline, professional inference system, and production-ready configuration management.*
