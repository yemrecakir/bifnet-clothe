#!/usr/bin/env python3
"""
BiRefNet API Server - Gradient Border Edition
FastAPI server for clothing background removal with gradient borders
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
from border_birefnet import BorderBiRefNet
from PIL import Image
import base64
from io import BytesIO
import time

app = FastAPI(
    title="BiRefNet Gradient Border API",
    description="🌈 Professional clothing background removal with gradient borders",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable - lazy loading to avoid startup timeout
border_processor = None
print("🚀 API server starting - BiRefNet will load on first request")

# Temp directory for processing
TEMP_DIR = Path("temp_api")
TEMP_DIR.mkdir(exist_ok=True)

def get_border_processor():
    """Lazy load BorderBiRefNet on first API request"""
    global border_processor
    
    if border_processor is None:
        print("🔄 Loading BiRefNet model (first request)...")
        from border_birefnet import BorderBiRefNet
        border_processor = BorderBiRefNet()
        print("✅ BiRefNet model loaded successfully!")
    
    return border_processor

@app.get("/")
async def root():
    return {
        "message": "🌈 BiRefNet Gradient Border API",
        "version": "1.0.0",
        "endpoints": {
            "POST /remove-background-simple": "🚀 Fast BiRefNet background removal",
            "POST /remove-background-base64": "Advanced processing with borders",
            "POST /remove-background": "File upload processing",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "loaded" if border_processor is not None else "ready_to_load",
        "startup": "fast"
    }

@app.post("/remove-background-simple")
async def remove_background_simple(request: dict):
    """
    🚀 Simple BiRefNet background removal - No item detection, no borders
    Just pure background removal for speed
    """
    
    if "image_base64" not in request:
        raise HTTPException(status_code=400, detail="image_base64 required")
    
    try:
        # Get BiRefNet processor (lazy loaded)
        processor = get_border_processor()
        
        # Decode base64 image
        print("📷 Decoding base64 image...")
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
            
            print(f"🚀 Simple background removal processing...")
            
            # Just do BiRefNet background removal - NO item detection, NO borders
            result_image, mask_image = processor.birefnet.remove_background(str(input_path))
            
            # Save result
            output_path = session_dir / "result.png"
            result_image.save(output_path, "PNG")
            
            # Convert to base64
            with open(output_path, "rb") as f:
                result_b64 = base64.b64encode(f.read()).decode()
            
            # Cleanup
            shutil.rmtree(session_dir, ignore_errors=True)
            
            print(f"✅ Simple background removal complete!")
            
            return {
                "success": True,
                "processing_type": "simple",
                "image_base64": result_b64
            }
            
        except Exception as e:
            # Cleanup on error
            shutil.rmtree(session_dir, ignore_errors=True)
            raise e
            
    except Exception as e:
        print(f"❌ Simple processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    border_type: str = Form("gradient"),
    border_width: int = Form(5),
    min_area: int = Form(2000)
):
    """
    🌈 Remove background with gradient borders
    
    Parameters:
    - image: Input image file
    - border_type: "gradient" or "solid" (default: gradient)
    - border_width: Border width in pixels (default: 5)
    - min_area: Minimum item area in pixels (default: 2000)
    """
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create unique session ID
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir()
    
    try:
        # Save uploaded image
        input_path = session_dir / f"input{Path(image.filename).suffix}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"🖼️ Processing: {image.filename} with {border_type} borders")
        
        # Process with BorderBiRefNet (lazy loaded)
        processor = get_border_processor()
        results = processor.detect_and_border_items(
            str(input_path),
            str(session_dir / "results"),
            border_type,
            border_width,
            min_area
        )
        
        if not results:
            raise HTTPException(status_code=400, detail="No clothing items detected")
        
        # Prepare response
        response_data = {
            "success": True,
            "session_id": session_id,
            "border_type": border_type,
            "border_width": border_width,
            "items_found": len(results),
            "items": []
        }
        
        # Process each item
        for item in results:
            item_data = {
                "id": int(item["id"]),  # Convert numpy.int32 to Python int
                "area": int(item["area"]),  # Convert numpy.int32 to Python int
                "full_image": f"/download/{session_id}/{Path(item['full_path']).name}",
                "cropped_image": f"/download/{session_id}/{Path(item['cropped_path']).name}"
            }
            response_data["items"].append(item_data)
        
        print(f"✅ Processed {len(results)} items with {border_type} borders")
        return response_data
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(session_dir, ignore_errors=True)
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/remove-background-base64")
async def remove_background_base64(
    request: dict,
    border_type: str = "gradient",
    border_width: int = 5,
    min_area: int = 2000
):
    """
    🚀 ULTRA FAST Process base64 encoded image with gradient borders - 320x320 resolution
    
    Request body:
    {
        "image_base64": "base64_string",
        "border_type": "gradient",
        "border_width": 5,
        "min_area": 2000
    }
    """
    
    if "image_base64" not in request:
        raise HTTPException(status_code=400, detail="image_base64 required")
    
    try:
        # Decode base64 image
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
        border_type = request.get("border_type", border_type)
        border_width = request.get("border_width", border_width)
        min_area = request.get("min_area", min_area)
        
        print(f"🚀 ULTRA FAST Processing base64 image with {border_type} borders at 320x320 resolution")
        start_time = time.time()
        
        # Process (lazy loaded) - Uses 320x320 optimization automatically
        processor = get_border_processor()
        results = processor.detect_and_border_items(
            str(input_path),
            str(session_dir / "results"),
            border_type,
            border_width,
            min_area
        )
        
        if not results:
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
                "id": int(item["id"]),  # Convert numpy.int32 to Python int
                "area": int(item["area"]),  # Convert numpy.int32 to Python int
                "cropped_image_base64": cropped_b64,
                "full_image_base64": full_b64
            })
        
        # Cleanup
        shutil.rmtree(session_dir, ignore_errors=True)
        
        processing_time = time.time() - start_time
        print(f"🚀 ULTRA FAST: Processed {len(results)} items in {processing_time:.2f} seconds with 320x320 optimization!")
        return {
            "success": True,
            "processing_time": f"{processing_time:.2f}s",
            "model": "BiRefNet 320x320 Ultra-Fast",
            "border_type": border_type,
            "border_width": border_width,
            "items_found": len(results),
            "items": response_items
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/remove-background-complex")
async def remove_background_complex(request: dict):
    """
    🚀 ULTRA FAST Complex Processing - 320x320 resolution for ~13 second processing
    Multiple item detection with gradient borders at maximum speed
    
    Request: {"image": "base64_string"}
    Response: {"success": true, "items": [...]}
    """
    if "image" not in request:
        raise HTTPException(status_code=400, detail="image field required")
    
    try:
        print("🔥 ULTRA FAST Complex Processing Started!")
        start_time = time.time()
        
        # Create session
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        results_dir = session_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode and save image
        image_data = base64.b64decode(request["image"])
        input_path = session_dir / "input.jpg"
        
        with open(input_path, "wb") as f:
            f.write(image_data)
        
        print(f"🎯 Processing: {input_path}")
        
        # Get processor and process with BorderBiRefNet (uses 320x320 optimization automatically)
        processor = get_border_processor()
        results = processor.detect_and_border_items(
            str(input_path),
            str(results_dir),
            border_type="gradient",
            border_width=5,
            min_area=2000
        )
        
        processing_time = time.time() - start_time
        print(f"⚡ ULTRA FAST processing completed in {processing_time:.2f} seconds!")
        
        if not results:
            raise HTTPException(status_code=400, detail="No clothing items detected")
        
        # Convert results to base64 for mobile compatibility
        response_items = []
        for item in results:
            # Read and encode images
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
        
        # Cleanup
        shutil.rmtree(session_dir, ignore_errors=True)
        
        print(f"🚀 ULTRA FAST Complex: {len(results)} items in {processing_time:.2f}s!")
        return {
            "success": True,
            "processing_time": f"{processing_time:.2f}s",
            "model": "BiRefNet 320x320 Ultra-Fast",
            "items": response_items
        }
        
    except Exception as e:
        # Cleanup on error
        if 'session_dir' in locals():
            shutil.rmtree(session_dir, ignore_errors=True)
        print(f"❌ ULTRA FAST Complex Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download processed image file"""
    
    file_path = TEMP_DIR / session_id / "results" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='image/png',
        filename=filename
    )

@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session files"""
    
    session_dir = TEMP_DIR / session_id
    
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return {"message": f"Session {session_id} cleaned up"}
    
    return {"message": "Session not found"}

# Startup cleanup
@app.on_event("startup")
async def startup_cleanup():
    """Clean temp directory on startup"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    print("🧹 Temp directory cleaned")

if __name__ == "__main__":
    print("🌈 Starting BiRefNet Gradient Border API Server...")
    print("📝 Endpoints:")
    print("   POST /remove-background - File upload")
    print("   POST /remove-background-base64 - Base64 processing")
    print("   GET /download/{session_id}/{filename} - Download results")
    print("🎯 Ready for gradient border magic!")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )