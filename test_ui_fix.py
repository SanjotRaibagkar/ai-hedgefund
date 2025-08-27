#!/usr/bin/env python3
"""
Test UI Fix
Simple test to verify the UI is working without layout errors.
"""

import requests
import time

def test_ui_access():
    """Test if the UI is accessible and working."""
    
    print("🧪 Testing UI Access")
    print("=" * 30)
    
    try:
        # Test basic access
        print("1. Testing basic UI access...")
        response = requests.get("http://localhost:8050", timeout=10)
        
        if response.status_code == 200:
            print("   ✅ UI is accessible")
            
            # Check for common error indicators in the response
            content = response.text.lower()
            
            if "error" in content and ("layout" in content or "component" in content):
                print("   ⚠️  Potential layout errors detected in response")
            else:
                print("   ✅ No obvious layout errors detected")
                
            if "dash" in content and "callback" in content:
                print("   ✅ Dash framework is working")
            else:
                print("   ⚠️  Dash framework might not be loading properly")
                
        else:
            print(f"   ❌ UI returned status code: {response.status_code}")
            
        print("\n2. Testing component availability...")
        
        # Test if key components are accessible
        components_to_test = [
            "chat-input",
            "chat-output", 
            "eod-results",
            "btn-init-chat"
        ]
        
        for component in components_to_test:
            if f'id="{component}"' in response.text:
                print(f"   ✅ Component '{component}' found in layout")
            else:
                print(f"   ❌ Component '{component}' missing from layout")
        
        print("\n🎉 UI Test Completed!")
        print("🌐 Access the UI at: http://localhost:8050")
        print("💡 If you see any errors, they should now be resolved!")
        
    except requests.exceptions.ConnectionError:
        print("   ❌ Could not connect to UI server")
        print("   💡 Make sure the server is running: poetry run python src/ui/web_app/app.py")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    test_ui_access()
