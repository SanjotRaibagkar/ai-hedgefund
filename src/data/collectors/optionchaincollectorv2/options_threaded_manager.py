#!/usr/bin/env python3
"""
Options Threaded Manager
Manages both collection and batch processing threads for options data.
Uses threading instead of multiprocessing to avoid DuckDB pickling issues.
"""

import threading
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from src.data.collectors.optionchaincollectorv2.options_chain_collector_v2 import OptionsChainCollectorV2

class OptionsThreadedManager:
    """Manages options data collection and batch processing threads."""
    
    def __init__(self):
        """Initialize the threaded manager."""
        self.logger = logging.getLogger(__name__)
        self.collector = OptionsChainCollectorV2()
        
        # Thread management
        self.collection_thread = None
        self.batch_thread = None
        self.running = False
        self.stop_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚀 Options Threaded Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.stop_all_threads()
        sys.exit(0)
    
    def start_collection_thread(self) -> bool:
        """Start the collection thread (Thread 1)."""
        try:
            if self.collection_thread is None or not self.collection_thread.is_alive():
                self.logger.info("🚀 Starting Options Collection Thread (Thread 1)")
                
                self.collection_thread = threading.Thread(
                    target=self._run_collection_thread,
                    name="OptionsCollection",
                    daemon=True
                )
                self.collection_thread.start()
                
                self.logger.info(f"✅ Collection thread started")
                return True
            else:
                self.logger.warning("⚠️ Collection thread is already running")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting collection thread: {e}")
            return False
    
    def start_batch_thread(self) -> bool:
        """Start the batch processing thread (Thread 2)."""
        try:
            if self.batch_thread is None or not self.batch_thread.is_alive():
                self.logger.info("🔄 Starting Options Batch Processing Thread (Thread 2)")
                
                self.batch_thread = threading.Thread(
                    target=self._run_batch_thread,
                    name="OptionsBatchProcessing",
                    daemon=True
                )
                self.batch_thread.start()
                
                self.logger.info(f"✅ Batch thread started")
                return True
            else:
                self.logger.warning("⚠️ Batch thread is already running")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting batch thread: {e}")
            return False
    
    def start_all_threads(self) -> Dict[str, bool]:
        """Start both collection and batch processing threads."""
        self.logger.info("🚀 Starting all options threads...")
        
        results = {}
        
        # Start collection thread
        results['collection'] = self.start_collection_thread()
        
        # Wait a moment for collection thread to initialize
        time.sleep(2)
        
        # Start batch thread
        results['batch'] = self.start_batch_thread()
        
        if results['collection'] and results['batch']:
            self.running = True
            self.logger.info("✅ All options threads started successfully")
        else:
            self.logger.error("❌ Failed to start all threads")
        
        return results
    
    def stop_collection_thread(self) -> bool:
        """Stop the collection thread."""
        try:
            if self.collection_thread and self.collection_thread.is_alive():
                self.logger.info("🛑 Stopping collection thread...")
                self.stop_event.set()
                self.collection_thread.join(timeout=10)
                
                if self.collection_thread.is_alive():
                    self.logger.warning("⚠️ Collection thread didn't stop gracefully")
                
                self.logger.info("✅ Collection thread stopped")
                return True
            else:
                self.logger.info("ℹ️ Collection thread is not running")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error stopping collection thread: {e}")
            return False
    
    def stop_batch_thread(self) -> bool:
        """Stop the batch processing thread."""
        try:
            if self.batch_thread and self.batch_thread.is_alive():
                self.logger.info("🛑 Stopping batch thread...")
                self.stop_event.set()
                self.batch_thread.join(timeout=10)
                
                if self.batch_thread.is_alive():
                    self.logger.warning("⚠️ Batch thread didn't stop gracefully")
                
                self.logger.info("✅ Batch thread stopped")
                return True
            else:
                self.logger.info("ℹ️ Batch thread is not running")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error stopping batch thread: {e}")
            return False
    
    def stop_all_threads(self) -> Dict[str, bool]:
        """Stop all threads."""
        self.logger.info("🛑 Stopping all options threads...")
        
        results = {}
        results['collection'] = self.stop_collection_thread()
        results['batch'] = self.stop_batch_thread()
        
        self.running = False
        self.stop_event.set()
        
        if results['collection'] and results['batch']:
            self.logger.info("✅ All threads stopped successfully")
        else:
            self.logger.warning("⚠️ Some threads may not have stopped properly")
        
        return results
    
    def get_thread_status(self) -> Dict[str, Any]:
        """Get status of all threads."""
        status = {
            'running': self.running,
            'collection_thread': {
                'alive': self.collection_thread.is_alive() if self.collection_thread else False,
                'name': self.collection_thread.name if self.collection_thread else None
            },
            'batch_thread': {
                'alive': self.batch_thread.is_alive() if self.batch_thread else False,
                'name': self.batch_thread.name if self.batch_thread else None
            }
        }
        
        return status
    
    def _run_collection_thread(self):
        """Run the collection thread (Thread 1)."""
        try:
            # Setup logging for the thread
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            
            logger = logging.getLogger(__name__)
            logger.info("🚀 Options Collection Thread (Thread 1) started")
            
            # Run the collection loop with stop event check
            while not self.stop_event.is_set():
                try:
                    if self.collector._is_market_hours():
                        # Collect data
                        results = self.collector.collect_options_data()
                        
                        # Log results
                        for index, result in results.items():
                            if result.get('success'):
                                logger.info(f"✅ {index}: {result.get('rows', 0)} rows collected")
                            else:
                                logger.warning(f"⚠️ {index}: {result.get('error', 'Unknown error')}")
                    else:
                        logger.info("⏰ Outside market hours, waiting...")
                    
                    # Wait for next collection or stop event
                    if self.stop_event.wait(self.collector.collection_interval):
                        break
                        
                except Exception as e:
                    logger.error(f"❌ Error in collection loop: {e}")
                    if self.stop_event.wait(60):  # Wait 1 minute on error
                        break
            
            logger.info("🛑 Collection thread stopped")
            
        except Exception as e:
            logger.error(f"❌ Error in collection thread: {e}")
            raise
    
    def _run_batch_thread(self):
        """Run the batch processing thread (Thread 2)."""
        try:
            # Setup logging for the thread
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            
            logger = logging.getLogger(__name__)
            logger.info("🔄 Options Batch Processing Thread (Thread 2) started")
            
            # Run the batch processing loop with stop event check
            while not self.stop_event.is_set():
                try:
                    if self.collector._is_market_hours():
                        # Process parquet files to DuckDB
                        results = self.collector.process_parquet_to_duckdb()
                        
                        logger.info(f"📊 Batch processing: {results['processed']} files, "
                                   f"{results['total_rows']} rows, {results['errors']} errors")
                    else:
                        logger.info("⏰ Outside market hours, waiting...")
                    
                    # Wait for next batch processing or stop event
                    if self.stop_event.wait(self.collector.batch_interval):
                        break
                        
                except Exception as e:
                    logger.error(f"❌ Error in batch processing loop: {e}")
                    if self.stop_event.wait(300):  # Wait 5 minutes on error
                        break
            
            logger.info("🛑 Batch processing thread stopped")
            
        except Exception as e:
            logger.error(f"❌ Error in batch thread: {e}")
            raise
    
    def run_monitoring_loop(self):
        """Run monitoring loop to check thread health."""
        self.logger.info("📊 Starting thread monitoring loop...")
        
        while self.running and not self.stop_event.is_set():
            try:
                status = self.get_thread_status()
                
                # Check collection thread
                if not status['collection_thread']['alive']:
                    self.logger.warning("⚠️ Collection thread died, restarting...")
                    self.start_collection_thread()
                
                # Check batch thread
                if not status['batch_thread']['alive']:
                    self.logger.warning("⚠️ Batch thread died, restarting...")
                    self.start_batch_thread()
                
                # Log status every 5 minutes
                if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
                    self.logger.info(f"📊 Thread Status: Collection={status['collection_thread']['alive']}, "
                                   f"Batch={status['batch_thread']['alive']}")
                
                # Wait for 30 seconds or stop event
                if self.stop_event.wait(30):
                    break
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                if self.stop_event.wait(30):
                    break


def main():
    """Main function for testing and running the threaded manager."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    manager = OptionsThreadedManager()
    
    try:
        # Start all threads
        results = manager.start_all_threads()
        
        if results['collection'] and results['batch']:
            print("✅ All threads started successfully")
            print("📊 Thread Status:")
            status = manager.get_thread_status()
            print(f"   Collection Thread: {'🟢 Running' if status['collection_thread']['alive'] else '🔴 Stopped'}")
            print(f"   Batch Thread: {'🟢 Running' if status['batch_thread']['alive'] else '🔴 Stopped'}")
            
            # Run monitoring loop
            manager.run_monitoring_loop()
        else:
            print("❌ Failed to start all threads")
            print(f"   Collection: {'✅' if results['collection'] else '❌'}")
            print(f"   Batch: {'✅' if results['batch'] else '❌'}")
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        manager.stop_all_threads()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        manager.stop_all_threads()


if __name__ == "__main__":
    main()
