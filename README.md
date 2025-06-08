# Mini LLaMA

A minimal yet powerful LLaMA-style transformer implementation optimized for RTX 4000 Ada GPU training with massive datasets, professional inference capabilities, and production-ready API server.

## 🖥️ Hardware Specifications
- **GPU**: NVIDIA RTX 4000 Ada Generation (20GB VRAM)
- **RAM**: 32GB + 8GB swap (upgraded from 16GB)
- **CPU**: 24 cores
- **Optimized for**: Professional AI/ML training, large-scale language model development, and production API deployment

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
# Core dependencies
pip install -r requirements.txt

# API server dependencies (additional)
pip install fastapi[all] uvicorn supabase python-dotenv
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

### **.env** - API Server Environment Configuration
```
# Supabase Configuration (Required for API server)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
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

### **Local Inference (Direct Model Access)**
```
# Simple text generation
python inference.py --prompt "Once upon a time" --temperature 0.8

# Interactive chat mode
python inference.py --interactive

# Benchmark performance testing
python inference.py --benchmark --runs 5
```

### **Production API Server (Recommended for Applications)**

#### **Setup API Server**
1. **Configure Supabase Database**
   - Create a new project at [supabase.com](https://supabase.com)
   - Copy your project URL and API keys
   - Run the database setup SQL (see API Server section below)

2. **Setup Environment Variables**
   ```
   # Create .env file with your Supabase credentials
   cp .env.example .env
   # Edit .env with your actual Supabase URL and keys
   ```

3. **Start API Server**
   ```
   python api_server.py
   ```

4. **Create API Keys**
   ```
   # Create your first API key
   python manage_keys_supabase.py create --name "My App" --expires 365 --rate-limit 1000
   ```

#### **API Endpoints**

**Base URL**: `http://localhost:8000`

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/` | GET | API status and information | No |
| `/health` | GET | Health check | No |
| `/v1/generate` | POST | Generate text | Yes |
| `/v1/stream` | POST | Stream text generation | Yes |
| `/stats` | GET | Usage statistics | Yes |
| `/admin/create-key` | POST | Create new API key | No* |
| `/admin/keys` | GET | List all keys | No* |

*Admin endpoints should be secured in production

#### **API Usage Examples**

**cURL Example:**
```
curl -X POST "http://localhost:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-your-api-key-here" \
  -d '{
    "prompt": "Explain artificial intelligence",
    "max_length": 150,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }'
```

**JavaScript/React Integration:**
```
class MiniLlamaAPI {
    constructor(apiKey, baseURL = 'http://localhost:8000') {
        this.apiKey = apiKey;
        this.baseURL = baseURL;
    }

    async generateText(prompt, options = {}) {
        const response = await fetch(`${this.baseURL}/v1/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey
            },
            body: JSON.stringify({
                prompt,
                max_length: options.maxLength || 100,
                temperature: options.temperature || 0.8,
                top_k: options.topK || 50,
                top_p: options.topP || 0.9,
                repetition_penalty: options.repetitionPenalty || 1.1
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }

        return await response.json();
    }

    async getStats() {
        const response = await fetch(`${this.baseURL}/stats`, {
            headers: { 'X-API-Key': this.apiKey }
        });
        return await response.json();
    }
}

// Usage
const api = new MiniLlamaAPI('sk-your-api-key-here');
const result = await api.generateText("Hello AI assistant");
console.log(result.text);
```

**Python Client Example:**
```
import requests

class MiniLlamaClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    
    def generate(self, prompt, **kwargs):
        data = {
            "prompt": prompt,
            "max_length": kwargs.get("max_length", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
        }
        
        response = requests.post(
            f"{self.base_url}/v1/generate",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.json()}")

# Usage
client = MiniLlamaClient("sk-your-api-key-here")
result = client.generate("Explain quantum computing", max_length=200)
print(result["text"])
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

## 🔐 API Server Features

### **Supabase-Powered Backend**
- **Cloud-native database**: No local database setup required
- **Automatic scaling**: PostgreSQL database with auto-scaling
- **Real-time capabilities**: Built-in real-time subscriptions
- **Security**: Row Level Security and built-in authentication
- **Admin dashboard**: Supabase dashboard for data management

### **API Key Management**
```
# Create API keys
python manage_keys_supabase.py create --name "Frontend App" --expires 365 --rate-limit 1000

# List all keys
python manage_keys_supabase.py list

# Get usage statistics
python manage_keys_supabase.py stats --key-id llama_12345678
```

### **Security Features**
- ✅ **Custom API Key System**: Secure key generation with expiration
- ✅ **Rate Limiting**: Configurable requests per hour per key
- ✅ **Usage Tracking**: Complete analytics and monitoring
- ✅ **Request Logging**: IP, user agent, response times tracked
- ✅ **CORS Support**: Ready for frontend integration
- ✅ **Error Handling**: Professional error responses

### **Performance Monitoring**
- **Real-time metrics**: Tokens/sec, response times, memory usage
- **Usage analytics**: Total requests, success rates, token counts
- **Performance stats**: Min/max generation speeds, averages
- **Database logging**: All requests logged to Supabase

## 📈 Expected Performance

### **Model Specifications**
- **Parameters**: ~95M (optimal for 20GB VRAM)
- **Training Speed**: 800-1200 tokens/second
- **Generation Speed**: 40-60 tokens/second (local), 30-50 tokens/second (API)
- **Memory Usage**: 14-16GB VRAM during training, 8GB during inference
- **Context Length**: 1024 tokens

### **API Performance**
- **Response Time**: 1-3 seconds for 100 tokens
- **Throughput**: 20-30 requests/minute per API key
- **Concurrent Users**: 10-50 (depending on hardware)
- **Uptime**: 99%+ with proper deployment

### **Training Timeline**
- **Quick Test**: 30 minutes (10K sequences)
- **Initial Results**: 2-4 hours (100K sequences)  
- **Good Model**: 8-12 hours (500K sequences)
- **Production Model**: 2-7 days (10.8M sequences)

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

## 🎮 Usage Workflows

### **Local Development Workflow**
```
# 1. Train your model
python -m src.train

# 2. Test inference locally
python inference.py --interactive

# 3. Start API server for production
python api_server.py

# 4. Create API key and test
python manage_keys_supabase.py create --name "test"
curl -X POST "http://localhost:8000/v1/generate" \
  -H "X-API-Key: your-key" \
  -d '{"prompt": "Hello AI"}'
```

### **Production Deployment Workflow**
```
# 1. Setup environment
cp .env.example .env
# Edit .env with production Supabase credentials

# 2. Deploy API server
python api_server.py --host 0.0.0.0 --port 8000

# 3. Setup reverse proxy (nginx/apache)
# 4. Configure SSL/TLS certificates
# 5. Setup monitoring and logging
```

### **Frontend Integration Workflow**
```
# 1. Create API key for your app
python manage_keys_supabase.py create --name "My Web App" --rate-limit 5000

# 2. Integrate into frontend (React/Vue/Angular)
const api = new MiniLlamaAPI('your-api-key');

# 3. Handle responses and errors
# 4. Monitor usage via /stats endpoint
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
├── api_server.py              # Production API server (NEW)
├── manage_keys_supabase.py    # API key management (NEW)
├── stream_concatenate.py      # Checkpoint concatenation
├── .env                       # Environment variables (NEW)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🛠️ Database Setup (Supabase)

Run this SQL in your Supabase SQL Editor to setup the API key management tables:

```
-- Create API Keys table
CREATE TABLE api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    key_id TEXT UNIQUE NOT NULL,
    key_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit INTEGER DEFAULT 100,
    total_requests INTEGER DEFAULT 0,
    last_used TIMESTAMP WITH TIME ZONE
);

-- Create API Usage tracking table
CREATE TABLE api_usage (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    key_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms INTEGER,
    tokens_generated INTEGER,
    success BOOLEAN,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (key_id) REFERENCES api_keys(key_id)
);

-- Create indexes for performance
CREATE INDEX idx_api_keys_key_id ON api_keys(key_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_usage_key_id ON api_usage(key_id);

-- Enable Row Level Security
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_usage ENABLE ROW LEVEL SECURITY;

-- Create policies for service role access
CREATE POLICY "Enable all operations for service role" ON api_keys
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON api_usage
    FOR ALL USING (auth.role() = 'service_role');
```

## 📝 Dataset Statistics

Your preprocessed corpus:
- **Total Sequences**: 10,821,218
- **Total Tokens**: ~11 billion  
- **File Size**: 41.28 GB
- **Sequence Length**: 1024 tokens each
- **Format**: Memory-mapped binary (.dat)

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

### **Common API Issues**
```
# Supabase connection error
# Check .env file has correct SUPABASE_URL and SUPABASE_SERVICE_KEY

# API key invalid
# Verify API key is correctly formatted: sk-xxxxxxxxxxxx

# Rate limit exceeded
# Check current usage: python manage_keys_supabase.py stats --key-id your-key-id
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

## 📚 References & Resources

- **Supabase**: https://supabase.com/docs
- **FastAPI**: https://fastapi.tiangolo.com/
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
- **Scalable API architecture** with cloud-native backend
- **Professional API key management** with usage tracking
- **Frontend-ready integration** with comprehensive examples

---

*Last Updated: Based on complete implementation including advanced training pipeline, professional inference system, production-ready API server with Supabase integration, and comprehensive frontend integration examples.*

