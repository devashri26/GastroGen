import os
import sys
import time
import webbrowser
import subprocess

def start_gui():
    print("====================================================")
    print("         GastroLens Medical AI Dashboard            ")
    print("====================================================")
    
    # Check if uvicorn is installed
    try:
        import uvicorn
    except ImportError:
        print("ERROR: 'uvicorn' is not installed. Please install it using:")
        print("  pip install uvicorn")
        sys.exit(1)

    # Start FastAPI server on port 8000
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    
    print(f"\nLaunching FastAPI backend server...")
    print(f"Server will be available at: {url}")
    
    # Open the browser once the server is responsive
    try:
        import threading
        import urllib.request
        
        def open_browser():
            for _ in range(60):
                time.sleep(1.0)
                try:
                    with urllib.request.urlopen(url, timeout=1) as resp:
                        if resp.status == 200:
                            break
                except Exception:
                    continue
            print(f"Opening default browser at {url}...")
            webbrowser.open(url)
            
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    except Exception as e:
        print(f"Could not prepare browser auto-launcher: {e}")

    # Run Uvicorn directly
    try:
        uvicorn.run("app:app", host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\nGastroLens Dashboard stopped by user request.")
    except Exception as e:
        print(f"\nError running uvicorn server: {e}")

if __name__ == "__main__":
    start_gui()
