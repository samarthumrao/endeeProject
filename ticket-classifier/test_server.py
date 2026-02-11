import requests

# Test server
try:
    r = requests.get('http://localhost:8000/submit/')
    print(f'Status Code: {r.status_code}')
    if r.status_code == 200:
        print('βœ… Submit page loaded successfully!')
        print(f'Title present: {"Submit Support Ticket" in r.text}')
    else:
        print(f'❌ Error: {r.status_code}')
except Exception as e:
    print(f'❌ Connection error: {e}')
