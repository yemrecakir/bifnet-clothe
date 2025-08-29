#!/usr/bin/env python3
"""
Startup Warmup Script - Pre-warm models during container startup
Reduces cold start time by loading models in background
"""

import asyncio
import time
import threading
from pathlib import Path
from model_cache import preload_default_models, global_model_cache
import torch

class StartupWarmer:
    """Handles model preloading and warmup during startup"""
    
    def __init__(self):
        self.warmup_complete = False
        self.warmup_thread = None
        self.start_time = time.time()
        
    def start_warmup(self):
        """Start warmup process in background thread"""
        if self.warmup_thread is None:
            print("🔥 Starting background model warmup...")
            self.warmup_thread = threading.Thread(
                target=self._warmup_worker,
                name="startup-warmer",
                daemon=True
            )
            self.warmup_thread.start()
    
    def _warmup_worker(self):
        """Background worker to preload models"""
        try:
            print("🚀 Warmup worker started")
            
            # Preload default models
            preload_default_models()
            
            # Warmup with dummy inference
            self._dummy_inference()
            
            self.warmup_complete = True
            warmup_time = time.time() - self.start_time
            print(f"✅ Model warmup complete in {warmup_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Warmup failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _dummy_inference(self):
        """Run dummy inference to warm up model"""
        try:
            print("🎯 Running dummy inference for warmup...")
            
            # Import here to avoid circular imports
            from final_birefnet import FinalBiRefNet
            from PIL import Image
            import numpy as np
            
            # Create small dummy image
            dummy_image = Image.new('RGB', (256, 256), color='red')
            
            # Initialize model (should use cache)
            birefnet = FinalBiRefNet()
            
            # Run inference
            result, mask = birefnet.remove_background(dummy_image)
            
            print("✅ Dummy inference successful - model is warm")
            
            # Clean up
            del result, mask, birefnet
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ Dummy inference failed: {e}")
    
    def wait_for_warmup(self, timeout: float = 60.0) -> bool:
        """Wait for warmup to complete"""
        if self.warmup_complete:
            return True
        
        if self.warmup_thread:
            self.warmup_thread.join(timeout=timeout)
            return self.warmup_complete
        
        return False
    
    def get_warmup_status(self) -> dict:
        """Get warmup status"""
        return {
            "warmup_complete": self.warmup_complete,
            "warmup_running": self.warmup_thread is not None and self.warmup_thread.is_alive(),
            "elapsed_time": time.time() - self.start_time,
            "cache_stats": global_model_cache.get_cache_stats()
        }

# Global warmer instance
startup_warmer = StartupWarmer()

def start_background_warmup():
    """Start background warmup process"""
    startup_warmer.start_warmup()

def wait_for_warmup(timeout: float = 60.0) -> bool:
    """Wait for warmup to complete"""
    return startup_warmer.wait_for_warmup(timeout)

def get_warmup_status() -> dict:
    """Get warmup status"""
    return startup_warmer.get_warmup_status()

if __name__ == "__main__":
    print("🚀 Testing Startup Warmer")
    print("=" * 40)
    
    # Start warmup
    start_background_warmup()
    
    # Monitor progress
    while not startup_warmer.warmup_complete:
        status = get_warmup_status()
        print(f"⏳ Warmup in progress... {status['elapsed_time']:.1f}s")
        time.sleep(2)
    
    final_status = get_warmup_status()
    print(f"✅ Warmup complete! Total time: {final_status['elapsed_time']:.2f}s")
    print(f"📊 Cache stats: {final_status['cache_stats']}")