#!/usr/bin/env python3
"""
Options Process Manager
Manages both collection and batch processing processes for options data.
"""

import multiprocessing
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from src.data.collectors.optionchaincollectorv2.options_chain_collector_v2 import OptionsChainCollectorV2

class OptionsProcessManager:
    """Manages options data collection and batch processing processes."""
    
    def __init__(self):
        """Initialize the process manager."""
        self.logger = logging.getLogger(__name__)
        self.collector = OptionsChainCollectorV2()
        
        # Process management
        self.collection_process = None
        self.batch_process = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚀 Options Process Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.stop_all_processes()
        sys.exit(0)
    
    def start_collection_process(self) -> bool:
        """Start the collection process (Process 1)."""
        try:
            if self.collection_process is None or not self.collection_process.is_alive():
                self.logger.info("🚀 Starting Options Collection Process (Process 1)")
                
                self.collection_process = multiprocessing.Process(
                    target=self._run_collection_process,
                    name="OptionsCollection"
                )
                self.collection_process.start()
                
                self.logger.info(f"✅ Collection process started with PID: {self.collection_process.pid}")
                return True
            else:
                self.logger.warning("⚠️ Collection process is already running")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting collection process: {e}")
            return False
    
    def start_batch_process(self) -> bool:
        """Start the batch processing process (Process 2)."""
        try:
            if self.batch_process is None or not self.batch_process.is_alive():
                self.logger.info("🔄 Starting Options Batch Processing Process (Process 2)")
                
                self.batch_process = multiprocessing.Process(
                    target=self._run_batch_process,
                    name="OptionsBatchProcessing"
                )
                self.batch_process.start()
                
                self.logger.info(f"✅ Batch process started with PID: {self.batch_process.pid}")
                return True
            else:
                self.logger.warning("⚠️ Batch process is already running")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting batch process: {e}")
            return False
    
    def start_all_processes(self) -> Dict[str, bool]:
        """Start both collection and batch processing processes."""
        self.logger.info("🚀 Starting all options processes...")
        
        results = {}
        
        # Start collection process
        results['collection'] = self.start_collection_process()
        
        # Wait a moment for collection process to initialize
        time.sleep(2)
        
        # Start batch process
        results['batch'] = self.start_batch_process()
        
        if results['collection'] and results['batch']:
            self.running = True
            self.logger.info("✅ All options processes started successfully")
        else:
            self.logger.error("❌ Failed to start all processes")
        
        return results
    
    def stop_collection_process(self) -> bool:
        """Stop the collection process."""
        try:
            if self.collection_process and self.collection_process.is_alive():
                self.logger.info("🛑 Stopping collection process...")
                self.collection_process.terminate()
                self.collection_process.join(timeout=10)
                
                if self.collection_process.is_alive():
                    self.logger.warning("⚠️ Collection process didn't stop gracefully, forcing...")
                    self.collection_process.kill()
                    self.collection_process.join()
                
                self.logger.info("✅ Collection process stopped")
                return True
            else:
                self.logger.info("ℹ️ Collection process is not running")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error stopping collection process: {e}")
            return False
    
    def stop_batch_process(self) -> bool:
        """Stop the batch processing process."""
        try:
            if self.batch_process and self.batch_process.is_alive():
                self.logger.info("🛑 Stopping batch process...")
                self.batch_process.terminate()
                self.batch_process.join(timeout=10)
                
                if self.batch_process.is_alive():
                    self.logger.warning("⚠️ Batch process didn't stop gracefully, forcing...")
                    self.batch_process.kill()
                    self.batch_process.join()
                
                self.logger.info("✅ Batch process stopped")
                return True
            else:
                self.logger.info("ℹ️ Batch process is not running")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error stopping batch process: {e}")
            return False
    
    def stop_all_processes(self) -> Dict[str, bool]:
        """Stop all processes."""
        self.logger.info("🛑 Stopping all options processes...")
        
        results = {}
        results['collection'] = self.stop_collection_process()
        results['batch'] = self.stop_batch_process()
        
        self.running = False
        
        if results['collection'] and results['batch']:
            self.logger.info("✅ All processes stopped successfully")
        else:
            self.logger.warning("⚠️ Some processes may not have stopped properly")
        
        return results
    
    def get_process_status(self) -> Dict[str, Any]:
        """Get status of all processes."""
        status = {
            'running': self.running,
            'collection_process': {
                'alive': self.collection_process.is_alive() if self.collection_process else False,
                'pid': self.collection_process.pid if self.collection_process else None
            },
            'batch_process': {
                'alive': self.batch_process.is_alive() if self.batch_process else False,
                'pid': self.batch_process.pid if self.batch_process else None
            }
        }
        
        return status
    
    def _run_collection_process(self):
        """Run the collection process (Process 1)."""
        try:
            # Setup logging for the process
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            
            logger = logging.getLogger(__name__)
            logger.info("🚀 Options Collection Process (Process 1) started")
            
            # Run the collection loop
            self.collector.run_collection_loop()
            
        except Exception as e:
            logger.error(f"❌ Error in collection process: {e}")
            raise
    
    def _run_batch_process(self):
        """Run the batch processing process (Process 2)."""
        try:
            # Setup logging for the process
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            
            logger = logging.getLogger(__name__)
            logger.info("🔄 Options Batch Processing Process (Process 2) started")
            
            # Run the batch processing loop
            self.collector.run_batch_processing_loop()
            
        except Exception as e:
            logger.error(f"❌ Error in batch process: {e}")
            raise
    
    def run_monitoring_loop(self):
        """Run monitoring loop to check process health."""
        self.logger.info("📊 Starting process monitoring loop...")
        
        while self.running:
            try:
                status = self.get_process_status()
                
                # Check collection process
                if not status['collection_process']['alive']:
                    self.logger.warning("⚠️ Collection process died, restarting...")
                    self.start_collection_process()
                
                # Check batch process
                if not status['batch_process']['alive']:
                    self.logger.warning("⚠️ Batch process died, restarting...")
                    self.start_batch_process()
                
                # Log status every 5 minutes
                if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
                    self.logger.info(f"📊 Process Status: Collection={status['collection_process']['alive']}, "
                                   f"Batch={status['batch_process']['alive']}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)


def main():
    """Main function for testing and running the process manager."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    manager = OptionsProcessManager()
    
    try:
        # Start all processes
        results = manager.start_all_processes()
        
        if results['collection'] and results['batch']:
            print("✅ All processes started successfully")
            print("📊 Process Status:")
            status = manager.get_process_status()
            print(f"   Collection Process: {'🟢 Running' if status['collection_process']['alive'] else '🔴 Stopped'}")
            print(f"   Batch Process: {'🟢 Running' if status['batch_process']['alive'] else '🔴 Stopped'}")
            
            # Run monitoring loop
            manager.run_monitoring_loop()
        else:
            print("❌ Failed to start all processes")
            print(f"   Collection: {'✅' if results['collection'] else '❌'}")
            print(f"   Batch: {'✅' if results['batch'] else '❌'}")
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        manager.stop_all_processes()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        manager.stop_all_processes()


if __name__ == "__main__":
    main()
