import os
import sys
import time
import webbrowser
import urllib.request
import threading

def start_new_gui():
    print("====================================================")
    print("   GastroLens — Pretrained Model Studio (New GUI)   ")
    print("====================================================")
    
    # Check if uvicorn is installed
    try:
        import uvicorn
    except ImportError:
        print("ERROR: 'uvicorn' is not installed. Please install it using:")
        print("  py -m pip install uvicorn")
        sys.exit(1)

    port = 8000
    bind_host = "0.0.0.0"  # Listen on both IPv4 and localhost
    access_url = f"http://127.0.0.1:{port}/new-gui"
    health_url = f"http://127.0.0.1:{port}/api/info"
    
    print("\n[1/3] Loading PyTorch models & generative checkpoints...")
    print("      (This may take 15-25 seconds on CPU. Please wait...)")

    def wait_and_open_browser():
        server_ready = False
        # Wait up to 120 seconds for models to finish loading
        for attempt in range(120):
            time.sleep(1.0)
            try:
                with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                    if resp.status == 200:
                        server_ready = True
                        break
            except Exception:
                continue
        
        if server_ready:
            print(f"\n[3/3] Backend is LIVE! Opening default browser at:")
            print(f"      --> {access_url}")
            print("====================================================\n")
            try:
                webbrowser.open(access_url)
            except Exception as e:
                print(f"Could not auto-open browser: {e}. Please open {access_url} manually.")
        else:
            print(f"\n[!] Server is taking longer than usual to start.")
            print(f"    Once loaded, open: {access_url}")

    browser_thread = threading.Thread(target=wait_and_open_browser, daemon=True)
    browser_thread.start()

    # Start Uvicorn
    print(f"[2/3] Starting backend server on port {port}...")
    try:
        uvicorn.run("app:app", host=bind_host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\nGastroLens Model Studio stopped.")
    except Exception as e:
        print(f"\nError starting uvicorn server: {e}")

if __name__ == "__main__":
    start_new_gui()
