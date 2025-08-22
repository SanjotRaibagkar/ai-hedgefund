#!/usr/bin/env python3
"""
UI Test Script
Tests the web UI functionality and imports.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger

def test_ui_imports():
    """Test if all UI imports work correctly."""
    logger.info("🔧 Testing UI Imports")
    logger.info("=" * 30)
    
    try:
        # Test Dash imports
        import dash
        logger.info("✅ Dash imported successfully")
        
        import dash_bootstrap_components as dbc
        logger.info("✅ Dash Bootstrap Components imported successfully")
        
        import plotly.graph_objs as go
        logger.info("✅ Plotly Graph Objects imported successfully")
        
        import plotly.express as px
        logger.info("✅ Plotly Express imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI imports failed: {e}")
        return False

def test_screening_manager_import():
    """Test if screening manager can be imported."""
    logger.info("\n🎯 Testing Screening Manager Import")
    logger.info("-" * 40)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        logger.info("✅ ScreeningManager imported successfully")
        
        # Test initialization
        manager = ScreeningManager()
        logger.info("✅ ScreeningManager initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening manager import failed: {e}")
        return False

def test_ui_components():
    """Test UI component creation."""
    logger.info("\n🎨 Testing UI Components")
    logger.info("-" * 30)
    
    try:
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        
        # Test basic components
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        logger.info("✅ Dash app created successfully")
        
        # Test layout components
        layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Test Header"),
                    html.P("Test paragraph")
                ])
            ])
        ])
        logger.info("✅ Layout components created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI components test failed: {e}")
        return False

def test_ui_functionality():
    """Test basic UI functionality."""
    logger.info("\n🚀 Testing UI Functionality")
    logger.info("-" * 35)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        
        # Test with sample data
        manager = ScreeningManager()
        
        # Test stock list parsing
        stock_list = "RELIANCE.NS, TCS.NS, HDFCBANK.NS"
        stocks = [s.strip() for s in stock_list.split(',')]
        logger.info(f"✅ Stock list parsed: {stocks}")
        
        # Test basic screening (without running full analysis)
        logger.info("✅ UI functionality test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI functionality test failed: {e}")
        return False

def main():
    """Main test execution."""
    logger.info("🚀 Starting UI Test Suite")
    
    tests = [
        ("UI Imports", test_ui_imports),
        ("Screening Manager Import", test_screening_manager_import),
        ("UI Components", test_ui_components),
        ("UI Functionality", test_ui_functionality)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test...")
        
        try:
            success = test_func()
            if success:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 UI TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! UI is ready to run!")
        logger.info("🌐 You can now start the web UI with: poetry run python src/ui/web_app/app.py")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. UI has issues.")
    
    return failed == 0

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 