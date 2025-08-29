#!/usr/bin/env python3
"""
API Server Starter - Quick start script for BiRefNet API
"""

import subprocess
import sys
import os

def start_api():
    print("🌈 BiRefNet Gradient Border API Server")
    print("=" * 45)
    print("🚀 Starting server on http://localhost:8000")
    print("📝 Endpoints:")
    print("   GET  /                           - API info")
    print("   POST /remove-background         - File upload (multipart/form-data)")
    print("   POST /remove-background-base64  - Base64 processing")
    print("   GET  /download/{session}/{file} - Download results") 
    print("   GET  /health                    - Health check")
    print()
    print("🌈 Gradient border mode: ACTIVE!")
    print("💡 Use border_type='gradient' for smooth fade borders")
    print("💡 Use border_type='solid' for clean white borders")
    print()
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api_server:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Try: pip install fastapi uvicorn python-multipart")

if __name__ == "__main__":
    start_api()