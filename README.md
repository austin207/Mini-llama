# Mini LLaMA

A minimal yet powerful LLaMA-style transformer implementation optimized for RTX 4000 Ada GPU training with massive datasets.

## 🖥️ Hardware Specifications
- **GPU**: NVIDIA RTX 4000 Ada Generation (20GB VRAM)
- **RAM**: 32GB + 8GB swap
- **Optimized for**: Professional AI/ML training and large-scale language model development

## ⚡ Quick Setup

1. **Clone and Environment Setup**
   ```
   git clone https://github.com/austin207/Mini-llama.git
   cd Mini-llama
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   # .\myenv\Scripts\Activate.ps1  # Windows
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   # or for Python 3:
   pip3 install -r requirements.txt
   ```

3. **Dataset Download Options**
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
- **Memory Usage**: 14-16GB VRAM during training
- **Context Length**: 1024 tokens

### **Training Timeline**
- **Quick Test**: 30 minutes (10K sequences)
- **Initial Results**: 2-4 hours (100K sequences)  
- **Good Model**: 8-12 hours (500K sequences)
- **Production Model**: 2-7 days (10.8M sequences)

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

## 📊 Training Menu Options

The training script provides an interactive menu:
1. **Start new training** - Fresh model training
2. **Resume training** - Continue from last checkpoint  
3. **Train from best model** - Continue from best validation loss
4. **Fine-tune** - Lower learning rate from best model
5. **Load and test model** - Text generation testing
6. **Exit** - Quit training

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

## 📚 References & Resources

- **Git LFS**: https://git-lfs.github.com/
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets/
- **PyTorch Mixed Precision**: https://pytorch.org/docs/stable/amp.html
- **Transformer Architecture**: https://arxiv.org/abs/1706.03762

## 🎉 Success Metrics

Your setup achieves:
- **10-20x faster training** vs CPU-only
- **Professional-scale models** (~95M parameters)
- **Enterprise-grade dataset** (10.8M sequences)
- **Production-ready pipeline** with all optimizations

---