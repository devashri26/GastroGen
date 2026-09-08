import os
import sys
import time
import subprocess
import webbrowser
import urllib.request
import threading

def check_frontend_dist():
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
    dist_index = os.path.join(dist_dir, "index.html")
    if not os.path.exists(dist_index):
        print("[*] Frontend distribution bundle (frontend/dist) not found.")
        print("[*] Automatically building production frontend bundle with npm...")
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        try:
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, check=True)
            print("[+] Frontend build completed successfully!")
        except Exception as e:
            print(f"[-] Warning: Failed to build frontend automatically: {e}")
            print("    Please run 'npm run build' manually inside 'frontend/' directory.")

def main():
    print("===================================================================")
    print("   GastroGen — Generative AI for Gastrointestinal Health (v2.0)   ")
    print("   Production Deployment Server                                    ")
    print("===================================================================")

    # 1. Check uvicorn
    try:
        import uvicorn
    except ImportError:
        print("ERROR: 'uvicorn' is not installed. Please install it using:")
        print("  pip install \"uvicorn[standard]\"")
        sys.exit(1)

    # 2. Check frontend build
    check_frontend_dist()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    local_url = f"http://127.0.0.1:{port}"
    health_url = f"http://127.0.0.1:{port}/api/health"

    print(f"\n[1/2] Initializing PyTorch models & GPU pipeline...")
    print(f"      Listening on: http://{host}:{port}")
    print(f"      Interactive API docs: http://{host}:{port}/docs\n")

    # Auto-open browser thread
    def wait_and_open():
        for _ in range(90):
            time.sleep(1.0)
            try:
                with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                    if resp.status == 200:
                        print(f"\n[+] GastroGen Production Server is LIVE!")
                        print(f"    --> Web UI: {local_url}")
                        print("===================================================================\n")
                        webbrowser.open(local_url)
                        break
            except Exception:
                continue

    threading.Thread(target=wait_and_open, daemon=True).start()

    # 3. Launch Uvicorn
    try:
        uvicorn.run("app:app", host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n[+] GastroGen server gracefully stopped.")
    except Exception as e:
        print(f"\n[-] Server runtime error: {e}")

if __name__ == "__main__":
    main()
