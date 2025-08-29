#!/usr/bin/env python3
"""
Lightweight BiRefNet API Server - Fast startup for Cloud Run
Lazy loading models to avoid timeout
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import json
import uuid
from PIL import Image
import base64
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BiRefNet Lightweight API",
    description="🚀 Fast-startup clothing background removal",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable - loaded on first request
border_processor = None
TEMP_DIR = Path("temp_api")
TEMP_DIR.mkdir(exist_ok=True)

def get_model():
    """Lazy load the model on first request"""
    global border_processor
    
    if border_processor is None:
        logger.info("🔄 Loading BiRefNet model (first request)...")
        try:
            from border_birefnet import BorderBiRefNet
            border_processor = BorderBiRefNet()
            logger.info("✅ BiRefNet model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    return border_processor

@app.get("/")
async def root():
    return {
        "message": "🚀 BiRefNet Lightweight API - Fast Startup",
        "version": "1.0.0",
        "status": "ready",
        "model_loaded": border_processor is not None,
        "endpoints": {
            "POST /remove-background-base64": "Process base64 image",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": border_processor is not None,
        "startup": "fast"
    }

@app.post("/remove-background-base64")
async def remove_background_base64(request: dict):
    """
    🌈 Process base64 encoded image with gradient borders
    Model loads on first request to avoid startup timeout
    """
    
    if "image_base64" not in request:
        raise HTTPException(status_code=400, detail="image_base64 required")
    
    try:
        # Get model (loads on first request)
        processor = get_model()
        
        # Decode base64 image
        logger.info("📷 Decoding base64 image...")
        image_data = base64.b64decode(request["image_base64"])
        image = Image.open(BytesIO(image_data))
        
        # Create session
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir()
        
        # Save image
        input_path = session_dir / "input.png"
        image.save(input_path, "PNG")
        
        # Get parameters
        border_type = request.get("border_type", "gradient")
        border_width = request.get("border_width", 5)
        min_area = request.get("min_area", 2000)
        
        logger.info(f"🖼️ Processing with {border_type} borders...")
        
        # Process
        results = processor.detect_and_border_items(
            str(input_path),
            str(session_dir / "results"),
            border_type,
            border_width,
            min_area
        )
        
        if not results:
            # Cleanup
            shutil.rmtree(session_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="No clothing items detected")
        
        # Convert results to base64
        response_items = []
        for item in results:
            # Read and encode images
            with open(item["cropped_path"], "rb") as f:
                cropped_b64 = base64.b64encode(f.read()).decode()
            
            with open(item["full_path"], "rb") as f:
                full_b64 = base64.b64encode(f.read()).decode()
            
            response_items.append({
                "id": item["id"],
                "area": item["area"],
                "cropped_image_base64": cropped_b64,
                "full_image_base64": full_b64
            })
        
        # Cleanup
        shutil.rmtree(session_dir, ignore_errors=True)
        
        logger.info(f"✅ Processed {len(results)} items successfully!")
        return {
            "success": True,
            "border_type": border_type,
            "border_width": border_width,
            "items_found": len(results),
            "items": response_items
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Startup cleanup
@app.on_event("startup")
async def startup_cleanup():
    """Fast startup - no model loading"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    logger.info("🚀 Lightweight API started - model will load on first request")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting on port {port}")
    
    uvicorn.run(
        "light_api_server:app",
        host="0.0.0.0",
        port=port
    )