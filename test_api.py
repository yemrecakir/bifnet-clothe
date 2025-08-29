#!/usr/bin/env python3
"""
API Test Script - Test the BiRefNet gradient border endpoints
"""

import requests
import base64
import json
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_file_upload(image_path="test2.jpg"):
    """Test file upload endpoint"""
    print(f"📤 Testing file upload with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image {image_path} not found!")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'border_type': 'gradient',
                'border_width': 5,
                'min_area': 2000
            }
            
            response = requests.post(f"{API_URL}/remove-background", files=files, data=data)
            
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Found {result['items_found']} items")
            print(f"🌈 Border type: {result['border_type']}")
            
            for item in result['items']:
                print(f"   📷 Item {item['id']}: {item['area']} pixels")
                print(f"      🔗 Full: {API_URL}{item['full_image']}")
                print(f"      ✂️  Crop: {API_URL}{item['cropped_image']}")
            
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return False

def test_base64_processing(image_path="test2.jpg"):
    """Test base64 endpoint"""
    print(f"🔢 Testing base64 processing with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image {image_path} not found!")
        return False
    
    try:
        # Convert to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        request_data = {
            "image_base64": image_data,
            "border_type": "gradient",
            "border_width": 5,
            "min_area": 2000
        }
        
        response = requests.post(f"{API_URL}/remove-background-base64", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Found {result['items_found']} items")
            print(f"🌈 Border type: {result['border_type']}")
            
            # Save first item as example
            if result['items']:
                first_item = result['items'][0]
                output_path = f"api_test_result_item_{first_item['id']}.png"
                
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(first_item['cropped_image_base64']))
                
                print(f"💾 Saved example result: {output_path}")
            
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Base64 test failed: {e}")
        return False

def main():
    print("🌈 BiRefNet API Test Suite")
    print("=" * 35)
    
    # Test sequence
    tests = [
        ("Health Check", test_health),
        ("File Upload", test_file_upload), 
        ("Base64 Processing", test_base64_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📝 {test_name}")
        print("-" * 20)
        success = test_func()
        results.append((test_name, success))
        print()
    
    # Summary
    print("📊 Test Results:")
    print("=" * 20)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for gradient border magic!")
    else:
        print("⚠️  Some tests failed. Check server status and try again.")

if __name__ == "__main__":
    main()