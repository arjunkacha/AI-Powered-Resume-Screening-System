import subprocess
import sys
import os
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server."""
    backend_dir = Path("backend")
    os.chdir(backend_dir)
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    """Run the Streamlit frontend server."""
    frontend_dir = Path("frontend")
    os.chdir(frontend_dir)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Create temp directory if it doesn't exist
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Run the servers
    import threading
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Run frontend in the main thread
    run_frontend() 