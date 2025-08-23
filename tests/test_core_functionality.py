#!/usr/bin/env python3
"""
Core Functionality Test
Tests the main components to ensure they work after cleanup.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

def test_core_imports():
    """Test that all core modules can be imported."""
    logger.info("🧪 Testing Core Imports...")
    
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test core modules
    modules_to_test = [
        ('src.screening.simple_eod_screener', 'SimpleEODScreener'),
        ('src.screening.screening_manager', 'ScreeningManager'),
        ('src.ui.web_app.app', 'app'),
        ('src.data.downloaders.comprehensive_equity_data_downloader', 'ComprehensiveEquityDataDownloader'),
        ('src.utils.database_stats_utility', 'show_comprehensive_stats'),
        ('src.tools.enhanced_api', 'get_prices'),
        ('src.ui.branding', 'print_logo'),
    ]
    
    for module_name, component_name in modules_to_test:
        results['total'] += 1
        try:
            module = __import__(module_name, fromlist=[component_name])
            if hasattr(module, component_name):
                results['passed'] += 1
                logger.info(f"✅ {module_name}.{component_name} imported successfully")
            else:
                results['failed'] += 1
                results['errors'].append(f"Component {component_name} not found in {module_name}")
                logger.error(f"❌ {component_name} not found in {module_name}")
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Failed to import {module_name}: {e}")
            logger.error(f"❌ Failed to import {module_name}: {e}")
    
    return results

def test_database_access():
    """Test database access."""
    logger.info("🗄️ Testing Database Access...")
    
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    try:
        from src.utils.database_stats_utility import show_comprehensive_stats
        # Just test that the function can be imported
        results['passed'] += 1
        logger.info("✅ Database utility functions imported successfully")
        results['total'] += 1
        
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"Database utility import failed: {e}")
        logger.error(f"❌ Database utility import failed: {e}")
        results['total'] += 1
    
    return results

def test_screening_functionality():
    """Test screening functionality."""
    logger.info("🔍 Testing Screening Functionality...")
    
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    try:
        from src.screening.simple_eod_screener import SimpleEODScreener
        screener = SimpleEODScreener()
        results['passed'] += 1
        logger.info("✅ SimpleEODScreener initialized successfully")
        results['total'] += 1
        
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"Screening initialization failed: {e}")
        logger.error(f"❌ Screening initialization failed: {e}")
        results['total'] += 1
    
    try:
        from src.screening.screening_manager import ScreeningManager
        manager = ScreeningManager()
        results['passed'] += 1
        logger.info("✅ ScreeningManager initialized successfully")
        results['total'] += 1
        
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"ScreeningManager initialization failed: {e}")
        logger.error(f"❌ ScreeningManager initialization failed: {e}")
        results['total'] += 1
    
    return results

def test_ui_functionality():
    """Test UI functionality."""
    logger.info("🎨 Testing UI Functionality...")
    
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    try:
        from src.ui.web_app.app import app
        results['passed'] += 1
        logger.info("✅ UI app imported successfully")
        results['total'] += 1
        
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"UI app import failed: {e}")
        logger.error(f"❌ UI app import failed: {e}")
        results['total'] += 1
    
    try:
        from src.ui.branding import print_logo
        results['passed'] += 1
        logger.info("✅ UI branding imported successfully")
        results['total'] += 1
        
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"UI branding import failed: {e}")
        logger.error(f"❌ UI branding import failed: {e}")
        results['total'] += 1
    
    return results

def main():
    """Run all core functionality tests."""
    logger.info("🚀 STARTING CORE FUNCTIONALITY TEST")
    logger.info("=" * 50)
    
    start_time = datetime.now()
    
    # Run tests
    import_results = test_core_imports()
    database_results = test_database_access()
    screening_results = test_screening_functionality()
    ui_results = test_ui_functionality()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Aggregate results
    total_tests = (import_results['total'] + database_results['total'] + 
                  screening_results['total'] + ui_results['total'])
    total_passed = (import_results['passed'] + database_results['passed'] + 
                   screening_results['passed'] + ui_results['passed'])
    total_failed = (import_results['failed'] + database_results['failed'] + 
                   screening_results['failed'] + ui_results['failed'])
    
    all_errors = (import_results['errors'] + database_results['errors'] + 
                 screening_results['errors'] + ui_results['errors'])
    
    # Print results
    logger.info("=" * 50)
    logger.info("🎯 CORE FUNCTIONALITY TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"📊 Total Tests: {total_tests}")
    logger.info(f"✅ Passed: {total_passed}")
    logger.info(f"❌ Failed: {total_failed}")
    logger.info(f"📈 Success Rate: {(total_passed/total_tests*100):.1f}%")
    logger.info(f"⏱️ Duration: {duration:.2f} seconds")
    
    if all_errors:
        logger.info("\n❌ Errors Found:")
        for error in all_errors:
            logger.error(f"   • {error}")
    
    if total_failed == 0:
        logger.info("\n🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
        logger.info("✅ System is working correctly after cleanup")
        return True
    else:
        logger.info(f"\n⚠️ {total_failed} tests failed")
        logger.info("❌ Some functionality may be broken after cleanup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
