"""
Test Endee API endpoints to verify connectivity and correct usage.
"""

import requests
import json

# Test base URL
base_url = "http://localhost:8080"

print("Testing Endee API Endpoints")
print("=" * 80)

# 1. Test health/root endpoint
print("\n1. Testing root endpoint...")
try:
    r = requests.get(f"{base_url}/")
    print(f"   Status: {r.status_code}")
except Exception as e:
    print(f"   Error: {e}")

# 2. List indexes
print("\n2. Listing indexes...")
try:
    r = requests.get(f"{base_url}/api/v1/index/list")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text}")
except Exception as e:
    print(f"   Error: {e}")

# 3. Try to create an index (test)
print("\n3. Creating test index...")
try:
    payload = {
        "dimension": 384,
        "metric": "cosine"
    }
    r = requests.post(
        f"{base_url}/api/v1/index/test_index",
        json=payload,
        headers={"Content-Type": "application/json"}  
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text if r.text else 'No content'}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("API Test Complete")
