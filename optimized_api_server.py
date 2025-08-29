#!/usr/bin/env python3
"""
Optimized BiRefNet API Server - Performance Enhanced
Optimized for Google Cloud Run with minimal startup time and maximum throughput
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import json
import uuid
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import gc

from border_birefnet import BorderBiRefNet
from PIL import Image
import base64
from io import BytesIO
import torch
from startup_warmup import start_background_warmup, get_warmup_status

# FastAPI app with performance optimizations
app = FastAPI(
    title="BiRefNet Optimized API",
    description="🚀 Ultra-fast clothing background removal optimized for Cloud Run",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
border_processor = None
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="birefnet")

# Temp directory for processing
TEMP_DIR = Path("temp_api")
TEMP_DIR.mkdir(exist_ok=True)

# Performance monitoring
request_count = 0
total_processing_time = 0.0

class ModelManager:
    """Thread-safe model manager with warm-up"""
    
    def __init__(self):
        self.model = None
        self.loading = False
        self.load_event = threading.Event()
    
    def get_model(self):
        """Get model with thread-safe lazy loading"""
        if self.model is not None:
            return self.model
        
        with model_lock:
            if self.model is not None:
                return self.model
            
            if not self.loading:
                self.loading = True
                print("🔄 Loading optimized BiRefNet model...")
                start_time = time.time()
                
                try:
                    self.model = BorderBiRefNet()
                    load_time = time.time() - start_time
                    print(f"✅ BiRefNet loaded in {load_time:.2f}s with optimizations!")
                    self.load_event.set()
                except Exception as e:
                    print(f"❌ Model loading failed: {e}")
                    self.loading = False
                    raise e
            else:
                # Wait for model to load
                self.load_event.wait(timeout=60)
                if self.model is None:
                    raise HTTPException(status_code=503, detail="Model loading timeout")
        
        return self.model

model_manager = ModelManager()

async def process_image_async(image_data: bytes, session_id: str, border_type: str = "gradient", 
                            border_width: int = 5, min_area: int = 2000):
    """Async wrapper for image processing"""
    loop = asyncio.get_event_loop()
    
    def sync_process():
        try:
            # Get model
            processor = model_manager.get_model()
            
            # Decode image
            image = Image.open(BytesIO(image_data))
            
            # Create session directory
            session_dir = TEMP_DIR / session_id
            session_dir.mkdir(exist_ok=True)
            results_dir = session_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            # Save input image
            input_path = session_dir / "input.png"
            image.save(input_path, "PNG")
            
            print(f"🚀 Processing {session_id} with optimized pipeline...")
            
            # Process with optimized settings
            results = processor.detect_and_border_items(
                str(input_path),
                str(results_dir),
                border_type,
                border_width,
                min_area
            )
            
            if not results:
                raise ValueError("No clothing items detected")
            
            # Convert to base64 for response
            response_items = []
            for item in results:
                # Read images efficiently
                with open(item["cropped_path"], "rb") as f:
                    cropped_b64 = base64.b64encode(f.read()).decode()
                
                with open(item["full_path"], "rb") as f:
                    full_b64 = base64.b64encode(f.read()).decode()
                
                response_items.append({
                    "id": int(item["id"]),
                    "area": int(item["area"]),
                    "cropped_image_base64": cropped_b64,
                    "full_image_base64": full_b64
                })
            
            # Cleanup immediately
            shutil.rmtree(session_dir, ignore_errors=True)
            
            # Force garbage collection for memory efficiency
            gc.collect()
            
            return response_items
            
        except Exception as e:
            # Cleanup on error
            if 'session_dir' in locals():
                shutil.rmtree(session_dir, ignore_errors=True)
            raise e
    
    # Execute in thread pool
    return await loop.run_in_executor(executor, sync_process)

@app.get("/")
async def root():
    return {
        "message": "🚀 BiRefNet Optimized API v2.0",
        "version": "2.0.0",
        "optimizations": [
            "🔥 FP16 inference",
            "⚡ Torch compile",
            "🧠 Memory optimized",
            "🎯 256px processing",
            "🔄 Async processing",
            "🧹 Auto cleanup"
        ],
        "performance": {
            "requests_processed": request_count,
            "average_time": f"{total_processing_time / max(request_count, 1):.2f}s"
        }
    }

@app.get("/health")
async def health_check():
    model_loaded = model_manager.model is not None
    warmup_status = get_warmup_status()
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "warmup_complete": warmup_status["warmup_complete"],
        "optimizations_enabled": True,
        "memory_usage": f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "CPU mode",
        "cache_stats": warmup_status.get("cache_stats", {})
    }

@app.get("/warmup")
async def warmup():
    """Warm up the model to reduce cold start time"""
    try:
        model_manager.get_model()
        return {"status": "warmed_up", "message": "Model is ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")

@app.post("/remove-background-ultra-fast")
async def remove_background_ultra_fast(request: dict):
    """
    🚀 ULTRA FAST - Optimized for minimal latency
    Target: <30 seconds on Cloud Run
    """
    global request_count, total_processing_time
    
    if "image_base64" not in request:
        raise HTTPException(status_code=400, detail="image_base64 required")
    
    start_time = time.time()
    session_id = str(uuid.uuid4())
    
    try:
        print(f"🚀 ULTRA FAST processing started: {session_id}")
        
        # Decode image data
        image_data = base64.b64decode(request["image_base64"])
        
        # Get parameters with defaults optimized for speed
        border_type = request.get("border_type", "gradient")
        border_width = request.get("border_width", 3)  # Reduced for speed
        min_area = request.get("min_area", 1000)  # Lower threshold for speed
        
        # Process asynchronously
        results = await process_image_async(
            image_data, session_id, border_type, border_width, min_area
        )
        
        processing_time = time.time() - start_time
        request_count += 1
        total_processing_time += processing_time
        
        print(f"🚀 ULTRA FAST complete: {processing_time:.2f}s")
        
        return {
            "success": True,
            "processing_time": f"{processing_time:.2f}s",
            "model": "BiRefNet 256px Ultra-Fast v2",
            "optimizations": "FP16 + Compile + Memory",
            "session_id": session_id,
            "items": results
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ ULTRA FAST error after {error_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/remove-background-simple")
async def remove_background_simple(request: dict):
    """
    🚀 Simple BiRefNet background removal - No borders, pure speed
    """
    if "image_base64" not in request:
        raise HTTPException(status_code=400, detail="image_base64 required")
    
    start_time = time.time()
    
    try:
        # Get model
        processor = model_manager.get_model()
        
        # Decode base64 image
        image_data = base64.b64decode(request["image_base64"])
        image = Image.open(BytesIO(image_data))
        
        # Create temp session
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir()
        
        try:
            # Save input image
            input_path = session_dir / "input.png"
            image.save(input_path, "PNG")
            
            print(f"🚀 Simple processing: {session_id}")
            
            # Direct BiRefNet processing - no item detection
            result_image, mask_image = processor.birefnet.remove_background(str(input_path))
            
            # Save result
            output_path = session_dir / "result.png"
            result_image.save(output_path, "PNG")
            
            # Convert to base64
            with open(output_path, "rb") as f:
                result_b64 = base64.b64encode(f.read()).decode()
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "processing_time": f"{processing_time:.2f}s",
                "processing_type": "simple_optimized",
                "image_base64": result_b64
            }
            
        finally:
            # Always cleanup
            shutil.rmtree(session_dir, ignore_errors=True)
            gc.collect()
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Simple processing error after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    return {
        "requests_processed": request_count,
        "total_processing_time": f"{total_processing_time:.2f}s",
        "average_time": f"{total_processing_time / max(request_count, 1):.2f}s",
        "model_loaded": model_manager.model is not None,
        "memory_usage": {
            "gpu": f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A",
            "gpu_cached": f"{torch.cuda.memory_reserved() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A"
        }
    }

# Background task for memory cleanup
async def periodic_cleanup():
    """Periodic memory cleanup"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Periodic cleanup completed")

@app.on_event("startup")
async def startup_cleanup():
    """Optimized startup with background warmup"""
    # Clean temp directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    
    # Start cleanup task
    asyncio.create_task(periodic_cleanup())
    
    # Start background model warmup
    start_background_warmup()
    
    print("🚀 Optimized API server started")
    print("🔥 Background model warmup in progress")
    print("⚡ First request will be faster due to pre-warming")

@app.on_event("shutdown")
async def shutdown_cleanup():
    """Cleanup on shutdown"""
    executor.shutdown(wait=True)
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    print("🛑 API server shutdown complete")

if __name__ == "__main__":
    print("🚀 Starting Optimized BiRefNet API Server...")
    print("⚡ Performance optimizations enabled")
    print("🎯 Target: <30 seconds processing time")
    
    uvicorn.run(
        "optimized_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for performance
        workers=1  # Single worker for model sharing
    )