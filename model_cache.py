#!/usr/bin/env python3
"""
Model Cache Manager - Pre-warm and cache models for faster inference
Optimized for Google Cloud Run with persistent model storage
"""

import os
import time
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import torch
from transformers import AutoModelForImageSegmentation

class ModelCache:
    """Thread-safe model cache with disk persistence"""
    
    def __init__(self, cache_dir: str = "/app/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.memory_cache: Dict[str, Any] = {}
        self.cache_lock = threading.RLock()
        self.load_times: Dict[str, float] = {}
        
        print(f"📦 ModelCache initialized: {self.cache_dir}")
    
    def _get_cache_key(self, model_name: str, device: str, **kwargs) -> str:
        """Generate unique cache key for model configuration"""
        config_str = f"{model_name}_{device}_{sorted(kwargs.items())}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for model"""
        return self.cache_dir / f"model_{cache_key}.pkl"
    
    def save_model_to_disk(self, model, cache_key: str) -> bool:
        """Save model state dict to disk for persistence"""
        try:
            cache_path = self._get_cache_path(cache_key)
            
            # Save model state dict instead of full model for compatibility
            model_data = {
                'state_dict': model.state_dict(),
                'model_class': type(model).__name__,
                'timestamp': time.time(),
                'device': str(model.device) if hasattr(model, 'device') else 'cpu'
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"💾 Model cached to disk: {cache_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to cache model to disk: {e}")
            return False
    
    def load_model_from_disk(self, cache_key: str) -> Optional[Dict]:
        """Load model data from disk cache"""
        try:
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            # Check if cache is not too old (24 hours)
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > 24 * 3600:
                print(f"🗑️ Cache expired, removing: {cache_path}")
                cache_path.unlink()
                return None
            
            with open(cache_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"📂 Model loaded from disk cache: {cache_path}")
            return model_data
            
        except Exception as e:
            print(f"⚠️ Failed to load model from disk: {e}")
            return None
    
    def get_or_load_model(self, model_name: str, device: torch.device, 
                         optimize_for_inference: bool = True, **kwargs):
        """Get model from cache or load and cache it"""
        
        device_str = str(device)
        cache_key = self._get_cache_key(model_name, device_str, 
                                      optimize=optimize_for_inference, **kwargs)
        
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                print(f"🎯 Model found in memory cache: {cache_key}")
                return self.memory_cache[cache_key]
            
            # Check disk cache
            cached_data = self.load_model_from_disk(cache_key)
            if cached_data:
                try:
                    print(f"🔄 Restoring model from disk cache...")
                    start_time = time.time()
                    
                    # Load fresh model and restore state
                    model = AutoModelForImageSegmentation.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    # Restore cached state dict
                    model.load_state_dict(cached_data['state_dict'])
                    model.to(device)
                    model.eval()
                    
                    # Apply optimizations
                    if optimize_for_inference:
                        model = self._optimize_model(model, device)
                    
                    load_time = time.time() - start_time
                    self.load_times[cache_key] = load_time
                    
                    # Cache in memory
                    self.memory_cache[cache_key] = model
                    
                    print(f"✅ Model restored from cache in {load_time:.2f}s")
                    return model
                    
                except Exception as e:
                    print(f"⚠️ Failed to restore from cache: {e}")
                    # Fall through to fresh load
            
            # Load fresh model
            print(f"🔄 Loading fresh model: {model_name}")
            start_time = time.time()
            
            model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                low_cpu_mem_usage=True,
                **kwargs
            )
            
            model.to(device)
            model.eval()
            
            # Apply optimizations
            if optimize_for_inference:
                model = self._optimize_model(model, device)
            
            load_time = time.time() - start_time
            self.load_times[cache_key] = load_time
            
            # Cache in memory
            self.memory_cache[cache_key] = model
            
            # Cache to disk for next startup
            self.save_model_to_disk(model, cache_key)
            
            print(f"✅ Fresh model loaded in {load_time:.2f}s")
            return model
    
    def _optimize_model(self, model, device):
        """Apply inference optimizations to model"""
        print("⚡ Applying inference optimizations...")
        
        try:
            # Try torch.compile for PyTorch 2.0+
            if hasattr(torch, 'compile') and device.type == 'cuda':
                model = torch.compile(model, mode='reduce-overhead')
                print("✅ Model compiled with torch.compile")
        except Exception as e:
            print(f"⚠️ Compile failed: {e}")
        
        try:
            # Half precision for GPU
            if device.type == 'cuda':
                model = model.half()
                print("✅ Using half precision (FP16)")
        except Exception as e:
            print(f"⚠️ Half precision failed: {e}")
        
        return model
    
    def preload_models(self, model_configs: list):
        """Preload multiple models during startup"""
        print(f"🔄 Preloading {len(model_configs)} models...")
        
        for config in model_configs:
            try:
                model_name = config.get('name', 'ZhengPeng7/BiRefNet')
                device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                
                print(f"🔄 Preloading: {model_name}")
                self.get_or_load_model(model_name, device)
                
            except Exception as e:
                print(f"⚠️ Failed to preload {config}: {e}")
        
        print("✅ Model preloading complete")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            disk_files = list(self.cache_dir.glob("model_*.pkl"))
            
            return {
                "memory_cached_models": len(self.memory_cache),
                "disk_cached_models": len(disk_files),
                "cache_directory": str(self.cache_dir),
                "load_times": self.load_times.copy(),
                "total_cache_size_mb": sum(f.stat().st_size for f in disk_files) / (1024**2)
            }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear model cache"""
        with self.cache_lock:
            # Clear memory cache
            self.memory_cache.clear()
            self.load_times.clear()
            print("🧹 Memory cache cleared")
            
            if not memory_only:
                # Clear disk cache
                for cache_file in self.cache_dir.glob("model_*.pkl"):
                    cache_file.unlink()
                print("🧹 Disk cache cleared")

# Global cache instance
global_model_cache = ModelCache()

def get_cached_model(model_name: str = 'ZhengPeng7/BiRefNet', 
                    device: Optional[torch.device] = None,
                    optimize_for_inference: bool = True):
    """Convenience function to get cached model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return global_model_cache.get_or_load_model(
        model_name, device, optimize_for_inference
    )

def preload_default_models():
    """Preload default models for faster cold starts"""
    configs = [
        {
            'name': 'ZhengPeng7/BiRefNet',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    ]
    
    global_model_cache.preload_models(configs)

if __name__ == "__main__":
    print("🚀 Model Cache Manager")
    print("=" * 40)
    
    # Test cache
    print("🧪 Testing model cache...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model twice to test caching
    print("📥 First load (should be slow):")
    model1 = get_cached_model(device=device)
    
    print("📥 Second load (should be fast from memory):")
    model2 = get_cached_model(device=device)
    
    print(f"🔍 Models identical: {model1 is model2}")
    
    # Show stats
    stats = global_model_cache.get_cache_stats()
    print(f"📊 Cache stats: {stats}")
    
    print("✅ Cache test complete")