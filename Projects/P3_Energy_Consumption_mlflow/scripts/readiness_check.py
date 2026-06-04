#!/usr/bin/env python3
"""
Readiness check script for the Energy Consumption ML Pipeline serving API.
Exits with code 0 if the API is ready, 1 otherwise.
"""
import sys
import urllib.request
import urllib.error
import time

def check_readiness(url="http://localhost:8000/health", retries=5, delay=2):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"API is ready! Status: 200 OK")
                    return 0
        except urllib.error.URLError as e:
            print(f"Attempt {i+1}/{retries}: API not ready yet ({e}). Waiting {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"Attempt {i+1}/{retries}: Unexpected error ({e}). Waiting {delay}s...")
            time.sleep(delay)
    
    print("API is not ready after all retries.")
    return 1

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/health"
    sys.exit(check_readiness(url))
