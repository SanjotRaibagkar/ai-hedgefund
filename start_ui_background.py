#!/usr/bin/env python3
"""
UI Background Runner
Starts the AI Hedge Fund web UI in the background.
"""

import sys
import os
import time
import subprocess
import psutil
import signal
from datetime import datetime
from loguru import logger
import threading

# Add src to path
sys.path.append('./src')


class UIBackgroundRunner:
    """Background runner for the web UI."""
    
    def __init__(self):
        """Initialize the UI background runner."""
        self.backend_process = None
        self.frontend_process = None
        self.is_running = False
        self.log_file = "logs/ui_background.log"
        
        # Setup logging
        logger.remove()
        logger.add(
            self.log_file,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
        )
        
        logger.info("🚀 UI Background Runner initialized")
    
    def start_backend(self):
        """Start the FastAPI backend server."""
        try:
            if self.backend_process and self.backend_process.poll() is None:
                logger.info("ℹ️ Backend is already running")
                return
            
            # Start the FastAPI backend
            cmd = [
                sys.executable, 
                "-m", "uvicorn", 
                "app.backend.main:app", 
                "--reload", 
                "--host", "127.0.0.1", 
                "--port", "8000"
            ]
            
            self.backend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            logger.info(f"🚀 Backend started with PID: {self.backend_process.pid}")
            logger.info("🌐 Backend URL: http://127.0.0.1:8000")
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
    
    def start_frontend(self):
        """Start the frontend development server."""
        try:
            if self.frontend_process and self.frontend_process.poll() is None:
                logger.info("ℹ️ Frontend is already running")
                return
            
            # Check if we're in the app directory
            frontend_dir = os.path.join(os.getcwd(), "app", "frontend")
            if not os.path.exists(frontend_dir):
                logger.warning("⚠️ Frontend directory not found, skipping frontend start")
                return
            
            # Start the frontend (assuming it's a React/Vite app)
            cmd = ["npm", "run", "dev"]
            
            self.frontend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=frontend_dir
            )
            
            logger.info(f"🚀 Frontend started with PID: {self.frontend_process.pid}")
            logger.info("🌐 Frontend URL: http://localhost:5173")
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
    
    def stop_backend(self):
        """Stop the backend server."""
        try:
            if self.backend_process and self.backend_process.poll() is None:
                # Send termination signal
                self.backend_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.backend_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    self.backend_process.kill()
                    self.backend_process.wait()
                
                logger.info("🛑 Backend stopped")
            else:
                logger.info("ℹ️ Backend is not running")
                
        except Exception as e:
            logger.error(f"❌ Error stopping backend: {e}")
    
    def stop_frontend(self):
        """Stop the frontend server."""
        try:
            if self.frontend_process and self.frontend_process.poll() is None:
                # Send termination signal
                self.frontend_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.frontend_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    self.frontend_process.kill()
                    self.frontend_process.wait()
                
                logger.info("🛑 Frontend stopped")
            else:
                logger.info("ℹ️ Frontend is not running")
                
        except Exception as e:
            logger.error(f"❌ Error stopping frontend: {e}")
    
    def is_backend_running(self) -> bool:
        """Check if backend process is running."""
        try:
            if self.backend_process:
                return self.backend_process.poll() is None
            return False
        except Exception:
            return False
    
    def is_frontend_running(self) -> bool:
        """Check if frontend process is running."""
        try:
            if self.frontend_process:
                return self.frontend_process.poll() is None
            return False
        except Exception:
            return False
    
    def run_background_ui(self):
        """Run the UI in background."""
        try:
            logger.info("🔄 Starting UI in background...")
            
            # Start backend
            self.start_backend()
            
            # Wait a moment for backend to start
            time.sleep(3)
            
            # Start frontend
            self.start_frontend()
            
            self.is_running = True
            
            logger.info("✅ UI started successfully!")
            logger.info("🌐 Backend: http://127.0.0.1:8000")
            logger.info("🌐 Frontend: http://localhost:5173")
            logger.info("📝 Logs: logs/ui_background.log")
            logger.info("🛑 Press Ctrl+C to stop the UI")
            
            while self.is_running:
                try:
                    # Check if processes are still running
                    if not self.is_backend_running():
                        logger.warning("⚠️ Backend stopped unexpectedly, restarting...")
                        self.start_backend()
                    
                    if not self.is_frontend_running():
                        logger.warning("⚠️ Frontend stopped unexpectedly, restarting...")
                        self.start_frontend()
                    
                    # Sleep for 30 seconds before checking again
                    time.sleep(30)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in UI monitoring loop: {e}")
                    time.sleep(30)  # Continue after error
            
            # Cleanup
            self.stop_frontend()
            self.stop_backend()
            logger.info("🔚 UI background runner stopped")
            
        except Exception as e:
            logger.error(f"❌ Fatal error in UI background runner: {e}")
    
    def stop(self):
        """Stop the UI background runner."""
        self.is_running = False
        self.stop_frontend()
        self.stop_backend()


def main():
    """Main function."""
    print("🚀 AI Hedge Fund UI Background Runner")
    print("="*60)
    print("🌐 Backend: http://127.0.0.1:8000")
    print("🌐 Frontend: http://localhost:5173")
    print("📝 Logs: logs/ui_background.log")
    print("🛑 Press Ctrl+C to stop")
    print("="*60)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize and run the UI background runner
        runner = UIBackgroundRunner()
        runner.run_background_ui()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
