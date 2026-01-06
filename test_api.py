import requests
import json

# Test the /analyze endpoint
url = "http://localhost:8000/analyze"

# Use a commit from the training data
payload = {
    "repo_url": "pallets/click",
    "commit_sha": "cdab890e57a30a9f437b88ce9652f7bfce980c1f"  # Real commit from click
}

print("="*60)
print("Testing CodeForensics API - /analyze endpoint")
print("="*60)
print(f"\nRequest: POST {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload, timeout=120)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS!")
        print("\nResponse:")
        print(json.dumps(result, indent=2))
        
        print("\n" + "="*60)
        print(f"RISK PREDICTION: {result['risk_score']*100:.1f}% - {result['risk_level']}")
        print("="*60)
    else:
        print(f"\n❌ Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ Cannot connect to API. Is it running?")
    print("   Start with: uvicorn src.api.main:app --reload")
except Exception as e:
    print(f"\n❌ Error: {e}")
