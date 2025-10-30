"""
MLflow server management module.
Handles server lifecycle and configuration.
"""

import os
import time
import threading
import psutil


class MLflowServer:
    """Manage MLflow tracking server."""
    
    def __init__(self, host="0.0.0.0", port=5000):
        """
        Initialize MLflow server manager.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server_thread = None
    
    def start(self):
        """Start MLflow server in background thread."""
        print("\n" + "=" * 70)
        print("[Step 4] Starting MLflow Server")
        print("=" * 70)
        
        # Terminate existing processes
        self._cleanup_existing_processes()
        
        # Start server
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        print("[Info] Waiting for MLflow server to start...")
        time.sleep(15)
        
        print(f"[URL] http://localhost:{self.port}")
        print("[Info] MLflow ready")
    
    def _run_server(self):
        """Run MLflow server command."""
        os.system(
            f"mlflow server --host {self.host} --port {self.port} "
            f"--backend-store-uri sqlite:///mlflow.db "
            f"--default-artifact-root ./mlruns"
        )
    
    @staticmethod
    def _cleanup_existing_processes():
        """Terminate existing MLflow processes."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and 'mlflow' in ' '.join(cmdline):
                    proc.kill()
                    print("[Info] Terminated existing MLflow process")
            except:
                pass
        
        time.sleep(2)
