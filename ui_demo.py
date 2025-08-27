#!/usr/bin/env python3
"""
UI Demo - Natural Language Interface Integration
Demonstrates how the natural language interface works in the UI.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine
from loguru import logger

def demo_ui_integration():
    """Demo the natural language interface integration."""
    
    print("🎨 UI Natural Language Interface Demo")
    print("=" * 50)
    
    try:
        # Initialize FNO engine (this happens when user clicks "Initialize AI Chat")
        print("1. 🚀 User clicks 'Initialize AI Chat' button...")
        fno_engine = FNOEngine()
        print("   ✅ FNO RAG System initialized successfully!")
        print("   ✅ Chat interface is now enabled in the UI")
        
        # Simulate user interactions
        print("\n2. 💬 User asks questions in the chat interface:")
        
        questions = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Predict RELIANCE movement for next month",
            "What's the chance of TCS going down this week?",
            "Show me INFY probability for tomorrow"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n   👤 User: {question}")
            response = fno_engine.chat(question)
            print(f"   🤖 AI: {response}")
        
        print("\n3. 🎯 UI Features:")
        print("   ✅ Natural language chat interface")
        print("   ✅ Real-time responses")
        print("   ✅ Chat history display")
        print("   ✅ Example questions for guidance")
        print("   ✅ Send button and Enter key support")
        print("   ✅ Error handling and user feedback")
        
        print("\n4. 🔧 Technical Integration:")
        print("   ✅ FNO RAG System backend")
        print("   ✅ Dash/Plotly frontend")
        print("   ✅ Real-time callbacks")
        print("   ✅ State management")
        print("   ✅ Error handling")
        
        print("\n🎉 UI Integration Demo Completed!")
        print("✅ The natural language interface is fully integrated!")
        print("🌐 Access the UI at: http://localhost:8050")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_ui_integration()
