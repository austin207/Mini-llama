import torch
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import os
from pathlib import Path
import time
import statistics
import json

# Import your model and tokenizer
from src.models.llama import MiniLlamaModel
from src.tokenizer.llama_tokenizer import LlamaTokenizer

class PerformanceMonitor:
    """Track and display performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.prompt_times = []
        self.generation_times = []
        self.prompt_token_counts = []
        self.generated_token_counts = []
        self.total_times = []
        
    def add_measurement(self, prompt_time, generation_time, prompt_tokens, generated_tokens):
        self.prompt_times.append(prompt_time)
        self.generation_times.append(generation_time)
        self.prompt_token_counts.append(prompt_tokens)
        self.generated_token_counts.append(generated_tokens)
        self.total_times.append(prompt_time + generation_time)
    
    def get_stats(self):
        if not self.generation_times:
            return None
            
        # Calculate tokens per second for different phases
        prompt_tps = [tokens/time if time > 0 else 0 for tokens, time in zip(self.prompt_token_counts, self.prompt_times)]
        generation_tps = [tokens/time if time > 0 else 0 for tokens, time in zip(self.generated_token_counts, self.generation_times)]
        total_tps = [(p_tokens + g_tokens)/(p_time + g_time) if (p_time + g_time) > 0 else 0 
                     for p_tokens, g_tokens, p_time, g_time in 
                     zip(self.prompt_token_counts, self.generated_token_counts, self.prompt_times, self.generation_times)]
        
        return {
            'runs': len(self.generation_times),
            'avg_prompt_tps': statistics.mean(prompt_tps) if prompt_tps else 0,
            'avg_generation_tps': statistics.mean(generation_tps) if generation_tps else 0,
            'avg_total_tps': statistics.mean(total_tps) if total_tps else 0,
            'avg_prompt_time': statistics.mean(self.prompt_times) if self.prompt_times else 0,
            'avg_generation_time': statistics.mean(self.generation_times) if self.generation_times else 0,
            'avg_total_time': statistics.mean(self.total_times) if self.total_times else 0,
            'avg_prompt_tokens': statistics.mean(self.prompt_token_counts) if self.prompt_token_counts else 0,
            'avg_generated_tokens': statistics.mean(self.generated_token_counts) if self.generated_token_counts else 0,
            'min_generation_tps': min(generation_tps) if generation_tps else 0,
            'max_generation_tps': max(generation_tps) if generation_tps else 0,
        }

class LlamaInference:
    def __init__(self, inference_config_path=None, **kwargs):
        """
        Initialize the inference engine with config file support and performance monitoring
        
        Args:
            inference_config_path: Path to inference.yaml config file
            **kwargs: Override config values (model_path, config_path, tokenizer_path, device)
        """
        # Load inference configuration
        self.config = self.load_inference_config(inference_config_path)
        
        # Override with command line arguments if provided
        if 'model_path' in kwargs and kwargs['model_path']:
            self.config['model']['checkpoint_path'] = kwargs['model_path']
        if 'config_path' in kwargs and kwargs['config_path']:
            self.config['model']['config_path'] = kwargs['config_path']
        if 'tokenizer_path' in kwargs and kwargs['tokenizer_path']:
            self.config['model']['tokenizer_path'] = kwargs['tokenizer_path']
        if 'device' in kwargs and kwargs['device']:
            self.config['model']['device'] = kwargs['device']
        
        # Setup device
        device_config = self.config['model']['device']
        if device_config == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)
        
        self.monitor = PerformanceMonitor()
        
        print(f"🚀 Loading model on {self.device}")
        print(f"📋 Config: {inference_config_path or 'Command line args'}")
        
        # Load model configuration
        with open(self.config['model']['config_path'], 'r') as f:
            self.model_config = yaml.safe_load(f)['model']
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizer(self.config['model']['tokenizer_path'])
        print(f"✅ Tokenizer loaded (vocab_size: {self.tokenizer.vocab_size})")
        
        # Load model
        self.model = MiniLlamaModel(self.model_config).to(self.device)
        
        # Load checkpoint
        checkpoint_path = self.config['model']['checkpoint_path']
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📊 Checkpoint info: Step {checkpoint.get('global_step', 'Unknown')}, "
                  f"Loss {checkpoint.get('best_val_loss', 'Unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {total_params:,} parameters")
        print(f"📊 Model config: {self.model_config}")

    def load_inference_config(self, config_path=None):
        """Load inference configuration from YAML file"""
        if config_path is None:
            config_path = "configs/inference.yaml"
        
        if not os.path.exists(config_path):
            print(f"⚠️  Config file {config_path} not found, using defaults")
            return self.get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded config from {config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config {config_path}: {e}")
            print("🔄 Using default configuration")
            return self.get_default_config()

    def get_default_config(self):
        """Get default configuration if config file is missing"""
        return {
            'model': {
                'checkpoint_path': 'checkpoints/mini-llama-best_loss1.2347.pt',
                'config_path': 'configs/model.yaml',
                'tokenizer_path': 'src/tokenizer/llama.model',
                'device': 'auto'
            },
            'generation': {
                'max_length': 100,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'do_sample': True,
                'num_return_sequences': 1
            },
            'benchmark': {
                'runs_per_prompt': 3,
                'default_prompts': [
                    "Once upon a time",
                    "The future of artificial intelligence is",
                    "In a world where technology",
                    "The quick brown fox",
                    "Explain the concept of machine learning",
                    "Write a short story about",
                    "The most important thing in life is",
                    "Scientists recently discovered"
                ]
            },
            'monitoring': {
                'enable_timing': True,
                'show_memory_usage': True,
                'save_stats': True,
                'stats_file': 'logs/inference_stats.json'
            }
        }

    def get_generation_defaults(self):
        """Get default generation parameters from config"""
        return self.config.get('generation', {
            'max_length': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        })

    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def apply_repetition_penalty(self, logits, input_ids, penalty=1.0):
        """Apply repetition penalty to reduce repeated tokens"""
        if penalty == 1.0:
            return logits
        
        for batch_idx in range(input_ids.size(0)):
            for token in input_ids[batch_idx].unique():
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] /= penalty
                else:
                    logits[batch_idx, token] *= penalty
        
        return logits

    def apply_temperature(self, logits, temperature=1.0):
        """Apply temperature scaling to logits"""
        if temperature == 1.0:
            return logits
        return logits / temperature

    def top_k_sampling(self, logits, k=50):
        """Apply top-k sampling"""
        if k <= 0:
            return logits
        
        top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_values)
        
        return mask

    def top_p_sampling(self, logits, p=0.9):
        """Apply top-p (nucleus) sampling"""
        if p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        logits_processed = torch.full_like(logits, float('-inf'))
        logits_processed.scatter_(-1, sorted_indices, sorted_logits)
        
        return logits_processed

    def generate_with_timing(self, prompt, max_length=None, temperature=None, top_k=None, 
                            top_p=None, repetition_penalty=None, do_sample=None, verbose=True):
        """
        Generate text with detailed timing measurements and config defaults
        """
        # Use config defaults if parameters not specified
        defaults = self.get_generation_defaults()
        max_length = max_length or defaults.get('max_length', 100)
        temperature = temperature or defaults.get('temperature', 0.8)
        top_k = top_k or defaults.get('top_k', 50)
        top_p = top_p or defaults.get('top_p', 0.9)
        repetition_penalty = repetition_penalty or defaults.get('repetition_penalty', 1.1)
        do_sample = do_sample if do_sample is not None else defaults.get('do_sample', True)
        
        # Count prompt tokens
        prompt_tokens = self.count_tokens(prompt)
        
        # Tokenize input
        prompt_start = time.time()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        prompt_time = time.time() - prompt_start
        
        original_length = input_ids.size(1)
        generated_tokens = 0
        
        if verbose:
            print(f"🎯 Generating: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
            print(f"📝 Prompt: '{prompt}' ({prompt_tokens} tokens)")
        
        # Generation timing
        generation_start = time.time()
        
        with torch.no_grad():
            for step in range(max_length):
                step_start = time.time()
                
                # Forward pass
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :]
                
                # Apply sampling techniques
                next_token_logits = self.apply_repetition_penalty(
                    next_token_logits, input_ids, repetition_penalty
                )
                
                if do_sample:
                    next_token_logits = self.apply_temperature(next_token_logits, temperature)
                    
                    if top_k > 0:
                        next_token_logits = self.top_k_sampling(next_token_logits, top_k)
                    
                    if top_p < 1.0:
                        next_token_logits = self.top_p_sampling(next_token_logits, top_p)
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                generated_tokens += 1
                
                step_time = time.time() - step_start
                
                # Real-time speed monitoring for verbose mode
                if verbose and step % 10 == 0 and step > 0:
                    current_tps = step / (time.time() - generation_start)
                    print(f"⚡ Step {step}: {current_tps:.1f} tokens/sec", end='\r')
                
                # Check for EOS token
                if hasattr(self.tokenizer, 'eos_token_id'):
                    if next_tokens.item() == self.tokenizer.eos_token_id:
                        break
        
        generation_time = time.time() - generation_start
        total_time = prompt_time + generation_time
        
        # Decode result
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        
        # Calculate performance metrics
        prompt_tps = prompt_tokens / prompt_time if prompt_time > 0 else 0
        generation_tps = generated_tokens / generation_time if generation_time > 0 else 0
        total_tps = (prompt_tokens + generated_tokens) / total_time if total_time > 0 else 0
        
        # Store measurements
        self.monitor.add_measurement(prompt_time, generation_time, prompt_tokens, generated_tokens)
        
        # Performance report
        if verbose:
            print(f"\n🚀 Generated {generated_tokens} tokens in {generation_time:.2f}s")
            print(f"📊 Performance:")
            print(f"   Prompt Processing: {prompt_tps:.1f} tokens/sec ({prompt_time:.3f}s)")
            print(f"   Generation: {generation_tps:.1f} tokens/sec ({generation_time:.2f}s)")
            print(f"   Total: {total_tps:.1f} tokens/sec ({total_time:.2f}s)")
            if self.config.get('monitoring', {}).get('show_memory_usage', True) and torch.cuda.is_available():
                print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        return {
            'text': generated_text,
            'prompt_tokens': prompt_tokens,
            'generated_tokens': generated_tokens,
            'prompt_time': prompt_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'prompt_tps': prompt_tps,
            'generation_tps': generation_tps,
            'total_tps': total_tps
        }

    def continuous_benchmark(self, prompts=None, max_length=None, temperature=None, top_k=None, 
                           top_p=None, repetition_penalty=None, runs_per_prompt=None):
        """
        Run continuous benchmark testing with config defaults
        """
        # Use config defaults if not specified
        if prompts is None:
            prompts = self.config.get('benchmark', {}).get('default_prompts', [
                "Once upon a time",
                "The future of artificial intelligence is",
                "In a world where technology",
                "The quick brown fox",
                "Explain the concept of machine learning",
                "Write a short story about",
                "The most important thing in life is",
                "Scientists recently discovered"
            ])
        
        defaults = self.get_generation_defaults()
        benchmark_config = self.config.get('benchmark', {})
        
        runs_per_prompt = runs_per_prompt or benchmark_config.get('runs_per_prompt', 3)
        max_length = max_length or defaults.get('max_length', 100)
        temperature = temperature or defaults.get('temperature', 0.8)
        top_k = top_k or defaults.get('top_k', 50)
        top_p = top_p or defaults.get('top_p', 0.9)
        repetition_penalty = repetition_penalty or defaults.get('repetition_penalty', 1.1)
        
        print(f"\n🔄 Starting Continuous Benchmark")
        print(f"📋 {len(prompts)} prompts × {runs_per_prompt} runs = {len(prompts) * runs_per_prompt} total tests")
        print("=" * 80)
        
        self.monitor.reset()
        
        try:
            for run in range(runs_per_prompt):
                print(f"\n🏃 Run {run + 1}/{runs_per_prompt}")
                print("-" * 40)
                
                for i, prompt in enumerate(prompts):
                    print(f"\n📝 Prompt {i+1}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
                    
                    result = self.generate_with_timing(
                        prompt=prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        verbose=False
                    )
                    
                    print(f"⚡ {result['generation_tps']:.1f} tokens/sec | "
                          f"{result['generated_tokens']} tokens | "
                          f"{result['generation_time']:.2f}s")
                    
                    # Show generated text sample
                    generated_part = result['text'][len(prompt):].strip()
                    preview = generated_part[:100] + '...' if len(generated_part) > 100 else generated_part
                    print(f"📄 '{preview}'")
                
                # Show running stats
                stats = self.monitor.get_stats()
                if stats:
                    print(f"\n📊 Running Average ({stats['runs']} samples):")
                    print(f"   Generation: {stats['avg_generation_tps']:.1f} tokens/sec")
                    print(f"   Range: {stats['min_generation_tps']:.1f} - {stats['max_generation_tps']:.1f} tokens/sec")
        
        except KeyboardInterrupt:
            print(f"\n⏸️  Benchmark interrupted by user")
        
        # Final statistics
        stats = self.monitor.get_stats()
        if stats:
            print(f"\n🏁 Final Benchmark Results ({stats['runs']} total runs):")
            print("=" * 60)
            print(f"📈 Average Performance:")
            print(f"   Prompt Processing: {stats['avg_prompt_tps']:.1f} tokens/sec")
            print(f"   Generation: {stats['avg_generation_tps']:.1f} tokens/sec")
            print(f"   Total: {stats['avg_total_tps']:.1f} tokens/sec")
            print(f"\n⏱️  Average Timing:")
            print(f"   Prompt Time: {stats['avg_prompt_time']:.3f}s")
            print(f"   Generation Time: {stats['avg_generation_time']:.2f}s")
            print(f"   Total Time: {stats['avg_total_time']:.2f}s")
            print(f"\n📊 Average Token Counts:")
            print(f"   Prompt Tokens: {stats['avg_prompt_tokens']:.1f}")
            print(f"   Generated Tokens: {stats['avg_generated_tokens']:.1f}")
            print(f"\n🎯 Performance Range:")
            print(f"   Min Generation Speed: {stats['min_generation_tps']:.1f} tokens/sec")
            print(f"   Max Generation Speed: {stats['max_generation_tps']:.1f} tokens/sec")
            
            # Save stats if configured
            if self.config.get('monitoring', {}).get('save_stats', False):
                self.save_performance_stats(stats)

    def save_performance_stats(self, stats):
        """Save performance statistics to file"""
        stats_file = self.config.get('monitoring', {}).get('stats_file', 'logs/inference_stats.json')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        
        # Add timestamp to stats
        stats['timestamp'] = time.time()
        stats['config'] = self.config
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📁 Performance stats saved to {stats_file}")
        except Exception as e:
            print(f"⚠️  Failed to save stats: {e}")

    def interactive_chat(self):
        """Interactive chat interface with performance monitoring and config defaults"""
        print("\n🤖 Mini-LLaMA Interactive Chat with Performance Monitoring")
        print("Commands: /help, /params, /stats, /reset, /config, /quit")
        print("=" * 70)
        
        # Get default parameters from config
        defaults = self.get_generation_defaults()
        temperature = defaults.get('temperature', 0.8)
        top_k = defaults.get('top_k', 50)
        top_p = defaults.get('top_p', 0.9)
        repetition_penalty = defaults.get('repetition_penalty', 1.1)
        max_length = defaults.get('max_length', 100)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == '/help':
                    print("\n📋 Commands:")
                    print("  /params - Show current parameters")
                    print("  /stats - Show performance statistics")
                    print("  /reset - Reset performance statistics")
                    print("  /config - Show config file info")
                    print("  /temp <value> - Set temperature")
                    print("  /topk <value> - Set top-k")
                    print("  /topp <value> - Set top-p")
                    print("  /penalty <value> - Set repetition penalty")
                    print("  /length <value> - Set max length")
                    print("  /quit - Exit chat")
                    continue
                elif user_input.lower() == '/config':
                    print(f"\n📋 Configuration Info:")
                    print(f"   Model: {self.config['model']['checkpoint_path']}")
                    print(f"   Tokenizer: {self.config['model']['tokenizer_path']}")
                    print(f"   Device: {self.device}")
                    print(f"   Default Temperature: {defaults['temperature']}")
                    print(f"   Default Top-k: {defaults['top_k']}")
                    print(f"   Default Top-p: {defaults['top_p']}")
                    continue
                elif user_input.lower() == '/stats':
                    stats = self.monitor.get_stats()
                    if stats:
                        print(f"\n📊 Performance Statistics ({stats['runs']} interactions):")
                        print(f"   Average Generation Speed: {stats['avg_generation_tps']:.1f} tokens/sec")
                        print(f"   Average Total Speed: {stats['avg_total_tps']:.1f} tokens/sec")
                        print(f"   Average Generation Time: {stats['avg_generation_time']:.2f}s")
                        print(f"   Speed Range: {stats['min_generation_tps']:.1f} - {stats['max_generation_tps']:.1f} tokens/sec")
                    else:
                        print("📊 No performance data yet. Start chatting to collect statistics!")
                    continue
                elif user_input.lower() == '/reset':
                    self.monitor.reset()
                    print("🔄 Performance statistics reset!")
                    continue
                elif user_input.lower() == '/params':
                    print(f"\n📊 Current Parameters:")
                    print(f"   Temperature: {temperature}")
                    print(f"   Top-k: {top_k}")
                    print(f"   Top-p: {top_p}")
                    print(f"   Repetition Penalty: {repetition_penalty}")
                    print(f"   Max Length: {max_length}")
                    continue
                elif user_input.startswith('/temp '):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"✅ Temperature set to {temperature}")
                        continue
                    except:
                        print("❌ Invalid temperature value")
                        continue
                elif user_input.startswith('/topk '):
                    try:
                        top_k = int(user_input.split()[1])
                        print(f"✅ Top-k set to {top_k}")
                        continue
                    except:
                        print("❌ Invalid top-k value")
                        continue
                elif user_input.startswith('/topp '):
                    try:
                        top_p = float(user_input.split()[1])
                        print(f"✅ Top-p set to {top_p}")
                        continue
                    except:
                        print("❌ Invalid top-p value")
                        continue
                elif user_input.startswith('/penalty '):
                    try:
                        repetition_penalty = float(user_input.split()[1])
                        print(f"✅ Repetition penalty set to {repetition_penalty}")
                        continue
                    except:
                        print("❌ Invalid repetition penalty value")
                        continue
                elif user_input.startswith('/length '):
                    try:
                        max_length = int(user_input.split()[1])
                        print(f"✅ Max length set to {max_length}")
                        continue
                    except:
                        print("❌ Invalid max length value")
                        continue
                
                if not user_input:
                    continue
                
                # Generate response with timing
                result = self.generate_with_timing(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    verbose=False
                )
                
                print(f"\n🤖 Mini-LLaMA: {result['text']}")
                print(f"⚡ {result['generation_tps']:.1f} tokens/sec | "
                      f"{result['generated_tokens']} tokens | "
                      f"{result['generation_time']:.2f}s")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Mini-LLaMA Inference with Config File Support')
    
    # Config file option
    parser.add_argument('--inference_config', type=str, default='configs/inference.yaml',
                        help='Path to inference config YAML file')
    
    # Path overrides (optional - will override config file values)
    parser.add_argument('--model', type=str, default=None,
                        help='Override model checkpoint path from config')
    parser.add_argument('--config', type=str, default=None,
                        help='Override model config path from config')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Override tokenizer path from config')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device from config (cuda/cpu/auto)')
    
    # Generation parameters (optional - will override config defaults)
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Override temperature from config')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Override top_k from config')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Override top_p from config')
    parser.add_argument('--repetition_penalty', type=float, default=None,
                        help='Override repetition_penalty from config')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Override max_length from config')
    
    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive chat mode')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run continuous benchmark mode')
    parser.add_argument('--runs', type=int, default=None,
                        help='Override benchmark runs from config')
    
    args = parser.parse_args()
    
    # Initialize inference engine with config file
    inference = LlamaInference(
        inference_config_path=args.inference_config,
        model_path=args.model,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        device=args.device
    )
    
    if args.benchmark:
        inference.continuous_benchmark(
            runs_per_prompt=args.runs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
    elif args.interactive:
        # Interactive chat mode
        inference.interactive_chat()
        
    elif args.prompt:
        # Single prompt generation
        result = inference.generate_with_timing(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        print(f"\n📄 Generated Text:")
        print("=" * 60)
        print(result['text'])
        print("=" * 60)
        
    else:
        print("❌ Please provide --prompt, use --interactive, or --benchmark mode")
        print("Examples:")
        print("  python inference.py --prompt 'Hello world' --max_length 50")
        print("  python inference.py --interactive")
        print("  python inference.py --benchmark --runs 5")
        print("  python inference.py --inference_config configs/my_inference.yaml --interactive")

if __name__ == "__main__":
    main()
