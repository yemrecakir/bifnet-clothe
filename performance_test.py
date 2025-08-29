#!/usr/bin/env python3
"""
Performance Test Script - Test optimized BiRefNet performance
Compare before/after optimization results
"""

import time
import base64
import requests
import json
from pathlib import Path
from PIL import Image
import io
import statistics
from typing import List, Dict

class PerformanceTest:
    """Performance testing for BiRefNet API"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.test_results = []
    
    def prepare_test_image(self, size=(512, 512)) -> str:
        """Create test image and encode as base64"""
        # Create a simple test image
        image = Image.new('RGB', size, color='red')
        
        # Add some content to make it more realistic
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 400, 400], fill='blue', outline='green', width=5)
        draw.ellipse([200, 200, 300, 300], fill='yellow')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_b64
    
    def test_endpoint(self, endpoint: str, payload: dict, test_name: str) -> Dict:
        """Test a single endpoint and measure performance"""
        print(f"\n🧪 Testing {test_name}...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}{endpoint}",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('success', False)
                
                test_result = {
                    'test_name': test_name,
                    'endpoint': endpoint,
                    'response_time': response_time,
                    'success': success,
                    'status_code': response.status_code,
                    'processing_time': result.get('processing_time', 'N/A'),
                    'items_found': len(result.get('items', [])),
                    'model_info': result.get('model', 'Unknown')
                }
                
                print(f"✅ {test_name}: {response_time:.2f}s (Success: {success})")
                if 'processing_time' in result:
                    print(f"   Server processing: {result['processing_time']}")
                if 'items' in result:
                    print(f"   Items detected: {len(result['items'])}")
                
            else:
                test_result = {
                    'test_name': test_name,
                    'endpoint': endpoint,
                    'response_time': response_time,
                    'success': False,
                    'status_code': response.status_code,
                    'error': response.text[:200]
                }
                
                print(f"❌ {test_name}: Failed ({response.status_code}) in {response_time:.2f}s")
                print(f"   Error: {response.text[:100]}...")
            
        except Exception as e:
            response_time = time.time() - start_time
            test_result = {
                'test_name': test_name,
                'endpoint': endpoint,
                'response_time': response_time,
                'success': False,
                'error': str(e)
            }
            
            print(f"❌ {test_name}: Exception in {response_time:.2f}s - {str(e)}")
        
        self.test_results.append(test_result)
        return test_result
    
    def run_performance_tests(self, num_iterations: int = 3):
        """Run comprehensive performance tests"""
        print("🚀 BiRefNet Performance Test Suite")
        print("=" * 50)
        
        # Check if server is running
        try:
            health_response = requests.get(f"{self.api_url}/health", timeout=10)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Server is healthy")
                print(f"   Model loaded: {health_data.get('model_loaded', False)}")
                print(f"   Warmup complete: {health_data.get('warmup_complete', False)}")
            else:
                print(f"⚠️ Server health check failed: {health_response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return
        
        # Prepare test image
        print("\n📷 Preparing test image...")
        test_image_b64 = self.prepare_test_image()
        print(f"   Image size: {len(test_image_b64)} chars")
        
        # Test configurations
        tests = [
            {
                'name': 'Ultra Fast Endpoint',
                'endpoint': '/remove-background-ultra-fast',
                'payload': {
                    'image_base64': test_image_b64,
                    'border_type': 'gradient',
                    'border_width': 3,
                    'min_area': 1000
                }
            },
            {
                'name': 'Simple Endpoint',
                'endpoint': '/remove-background-simple',
                'payload': {
                    'image_base64': test_image_b64
                }
            },
            {
                'name': 'Complex Endpoint',
                'endpoint': '/remove-background-complex',
                'payload': {
                    'image': test_image_b64
                }
            }
        ]
        
        # Run tests multiple times
        print(f"\n🔄 Running {len(tests)} tests x {num_iterations} iterations...")
        
        for test_config in tests:
            iteration_times = []
            
            for i in range(num_iterations):
                print(f"\n--- {test_config['name']} - Iteration {i+1}/{num_iterations} ---")
                
                result = self.test_endpoint(
                    test_config['endpoint'],
                    test_config['payload'],
                    f"{test_config['name']} #{i+1}"
                )
                
                if result['success']:
                    iteration_times.append(result['response_time'])
                
                # Wait between tests to avoid overwhelming the server
                if i < num_iterations - 1:
                    time.sleep(1)
            
            # Calculate statistics for this test
            if iteration_times:
                avg_time = statistics.mean(iteration_times)
                min_time = min(iteration_times)
                max_time = max(iteration_times)
                
                print(f"\n📊 {test_config['name']} Summary:")
                print(f"   Average: {avg_time:.2f}s")
                print(f"   Min: {min_time:.2f}s")
                print(f"   Max: {max_time:.2f}s")
                
                if len(iteration_times) > 1:
                    std_dev = statistics.stdev(iteration_times)
                    print(f"   Std Dev: {std_dev:.2f}s")
        
        # Overall summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("🎯 PERFORMANCE TEST SUMMARY")
        print("=" * 50)
        
        successful_tests = [r for r in self.test_results if r['success']]
        failed_tests = [r for r in self.test_results if not r['success']]
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if successful_tests:
            response_times = [r['response_time'] for r in successful_tests]
            print(f"\n⚡ Response Time Statistics:")
            print(f"   Average: {statistics.mean(response_times):.2f}s")
            print(f"   Minimum: {min(response_times):.2f}s")
            print(f"   Maximum: {max(response_times):.2f}s")
            
            if len(response_times) > 1:
                print(f"   Std Dev: {statistics.stdev(response_times):.2f}s")
        
        # Performance targets
        print(f"\n🎯 Performance Targets:")
        under_30s = len([r for r in successful_tests if r['response_time'] < 30])
        under_60s = len([r for r in successful_tests if r['response_time'] < 60])
        
        if successful_tests:
            print(f"   Under 30s: {under_30s}/{len(successful_tests)} ({100*under_30s/len(successful_tests):.1f}%)")
            print(f"   Under 60s: {under_60s}/{len(successful_tests)} ({100*under_60s/len(successful_tests):.1f}%)")
        
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in failed_tests[:5]:  # Show first 5 failures
                print(f"   {test['test_name']}: {test.get('error', 'Unknown error')}")
        
        # Recommendations
        if successful_tests:
            avg_time = statistics.mean([r['response_time'] for r in successful_tests])
            if avg_time > 60:
                print(f"\n💡 RECOMMENDATIONS:")
                print(f"   - Average response time ({avg_time:.1f}s) exceeds target")
                print(f"   - Consider further optimizations")
                print(f"   - Check Cloud Run configuration")
            elif avg_time > 30:
                print(f"\n💡 RECOMMENDATIONS:")
                print(f"   - Performance is acceptable but could be improved")
                print(f"   - Consider CPU scaling or model optimizations")
            else:
                print(f"\n✅ EXCELLENT PERFORMANCE!")
                print(f"   - Average response time: {avg_time:.1f}s")
                print(f"   - Performance targets met!")
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'results': self.test_results
            }, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")

def main():
    print("🧪 BiRefNet Performance Testing Tool")
    print("=" * 40)
    
    # Configuration
    api_url = "http://localhost:8080"  # Change for deployed version
    iterations = 3
    
    print(f"🎯 Target API: {api_url}")
    print(f"🔄 Iterations per test: {iterations}")
    
    # Run tests
    tester = PerformanceTest(api_url)
    tester.run_performance_tests(iterations)
    
    # Save results
    tester.save_results()
    
    print("\n🏁 Performance testing complete!")

if __name__ == "__main__":
    main()