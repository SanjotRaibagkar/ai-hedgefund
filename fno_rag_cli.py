#!/usr/bin/env python3
"""
Enhanced FNO RAG CLI Interface
==============================

Simple command-line interface for the enhanced FNO RAG system.
Allows users to query the system with natural language and get intelligent responses.
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from loguru import logger

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from build_enhanced_vector_store import EnhancedFNOVectorStore


class FNORAGCLI:
    """CLI interface for the Enhanced FNO RAG system."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.vector_store = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO"
        )
    
    def initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            logger.info("🔧 Initializing Enhanced FNO Vector Store...")
            self.vector_store = EnhancedFNOVectorStore()
            
            # Try to load existing vector store
            if self.vector_store.load_vector_store():
                logger.info("✅ Vector store loaded successfully")
                return True
            else:
                logger.warning("⚠️ No existing vector store found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            return False
    
    def build_vector_store(self, start_date: str = None, end_date: str = None):
        """Build the vector store."""
        if not self.vector_store:
            self.initialize_vector_store()
        
        logger.info("🚀 Building Enhanced FNO Vector Store...")
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        logger.info(f"📅 Building for period: {start_date} to {end_date}")
        
        try:
            self.vector_store.build_vector_store(start_date, end_date)
            logger.info("✅ Vector store built successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Error building vector store: {e}")
            return False
    
    def process_query(self, query: str, top_k: int = 5):
        """Process a natural language query."""
        if not self.vector_store:
            if not self.initialize_vector_store():
                logger.error("❌ Cannot initialize vector store")
                return
        
        logger.info(f"🔍 Processing query: {query}")
        logger.info("-" * 60)
        
        try:
            # Get RAG response
            result = self.vector_store.query_rag_system(query, top_k)
            
            # Display results
            self.display_results(result)
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
    
    def display_results(self, result: dict):
        """Display query results in a formatted way."""
        query = result['query']
        similar_cases = result['similar_cases']
        llm_response = result['llm_response']
        
        # Display query
        print(f"\n📝 Query: {query}")
        print("=" * 60)
        
        # Display similar cases
        if similar_cases:
            print(f"\n📊 Similar Historical Cases (Top {len(similar_cases)}):")
            print("-" * 40)
            
            for i, case in enumerate(similar_cases, 1):
                print(f"{i}. {case['symbol']} ({case['date']})")
                print(f"   • Close Price: ₹{case['close_price']:,.2f}")
                print(f"   • Daily Return: {case['daily_return']:+.2f}%")
                print(f"   • PCR: {case['pcr']:.2f}")
                print(f"   • Implied Vol: {case['implied_volatility']:.1f}%")
                print(f"   • Next Day: {case['next_day_return']:+.2f}% ({case['next_day_direction']})")
                print(f"   • Similarity Score: {case['similarity_score']:.3f}")
                print()
        else:
            print("\n❌ No similar cases found")
        
        # Display LLM analysis
        print(f"\n🤖 AI Analysis:")
        print("-" * 40)
        print(llm_response)
        print("=" * 60)
    
    def interactive_mode(self):
        """Run interactive mode for continuous queries."""
        logger.info("🎯 Starting Interactive FNO RAG Mode")
        logger.info("=" * 60)
        logger.info("💡 Type your queries in natural language")
        logger.info("💡 Examples:")
        logger.info("   • 'How much can RELIANCE move tomorrow?'")
        logger.info("   • 'Find cases where NIFTY rose with high Put OI'")
        logger.info("   • 'Show me BANKNIFTY patterns with low PCR'")
        logger.info("💡 Type 'quit' or 'exit' to stop")
        logger.info("=" * 60)
        
        if not self.initialize_vector_store():
            logger.error("❌ Cannot start interactive mode - vector store not available")
            return
        
        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter your query: ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Process query
                self.process_query(query)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    
    def run_example_queries(self):
        """Run a set of example queries to demonstrate the system."""
        example_queries = [
            "How much can RELIANCE move tomorrow based on current FNO data?",
            "Find similar cases where NIFTY rose with high Put OI",
            "Show me cases where BANKNIFTY had low PCR and moved up",
            "What happens when there's long buildup in stock futures?",
            "Find patterns where implied volatility was high and price moved down",
            "Show me cases where short covering led to price increases"
        ]
        
        logger.info("🧪 Running Example Queries")
        logger.info("=" * 60)
        
        for i, query in enumerate(example_queries, 1):
            logger.info(f"\n📝 Example {i}: {query}")
            self.process_query(query)
            
            if i < len(example_queries):
                input("\nPress Enter to continue to next example...")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Enhanced FNO RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python fno_rag_cli.py --interactive
  
  # Build vector store
  python fno_rag_cli.py --build
  
  # Single query
  python fno_rag_cli.py --query "How much can RELIANCE move tomorrow?"
  
  # Run examples
  python fno_rag_cli.py --examples
  
  # Build with custom date range
  python fno_rag_cli.py --build --start-date 2024-01-01 --end-date 2024-06-30
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--build', '-b',
        action='store_true',
        help='Build the vector store'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Process a single query'
    )
    
    parser.add_argument(
        '--examples', '-e',
        action='store_true',
        help='Run example queries'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for building vector store (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for building vector store (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of similar cases to retrieve (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = FNORAGCLI()
    
    # Process arguments
    if args.build:
        success = cli.build_vector_store(args.start_date, args.end_date)
        if not success:
            sys.exit(1)
    
    if args.query:
        cli.process_query(args.query, args.top_k)
    
    elif args.examples:
        cli.run_example_queries()
    
    elif args.interactive:
        cli.interactive_mode()
    
    else:
        # Default to interactive mode if no other options
        cli.interactive_mode()


if __name__ == "__main__":
    main()
