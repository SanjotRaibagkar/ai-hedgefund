# Complete Setup Guide - AI Hedge Fund with Indian Market Integration

This guide will walk you through the complete setup process for the AI Hedge Fund system with Indian market integration.

## 📋 Prerequisites Check

Before starting, ensure you have the following installed:

### 1. Python 3.8+
```bash
python --version
# Should show Python 3.8 or higher
```

### 2. Git
```bash
git --version
# Should show Git version
```

### 3. Internet Connection
- Required for downloading dependencies and fetching market data

## 🛠️ Step-by-Step Installation

### Step 1: Clone the Repository
```bash
# Clone the repository
git clone https://github.com/SanjotRaibagkar/ai-hedgefund.git

# Navigate to the project directory
cd ai-hedgefund

# Verify you're in the correct directory
ls
# Should show files like README.md, pyproject.toml, src/, etc.
```

### Step 2: Install Poetry

#### Windows Installation:
```powershell
# Run in PowerShell as Administrator
powershell -Command "Invoke-Expression (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content"

# Add Poetry to PATH (if not automatically added)
$env:Path += ";C:\Users\$env:USERNAME\AppData\Roaming\Python\Scripts"

# Verify installation
poetry --version
```

#### macOS/Linux Installation:
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Reload shell or run
source ~/.bashrc  # or source ~/.zshrc

# Verify installation
poetry --version
```

### Step 3: Install Dependencies
```bash
# Install all Python dependencies
poetry install

# This may take a few minutes as it downloads and installs packages
```

### Step 4: Activate Virtual Environment
```bash
# Activate the Poetry virtual environment
poetry shell

# Verify you're in the virtual environment
# Your prompt should show (ai-hedgefund-...)
```

## ⚙️ Configuration Setup

### Step 1: Create Environment File
```bash
# Create .env file in the root directory
touch .env  # On Windows: echo. > .env
```

### Step 2: Configure Environment Variables
Edit the `.env` file and add the following (replace with your actual keys):

```bash
# API Keys (Optional - system works without these for basic functionality)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///ai_hedge_fund.db

# Logging Configuration
LOG_LEVEL=INFO

# Optional: Custom settings
CACHE_TTL=300
MAX_RETRIES=3
```

**Note**: The system works without API keys for basic functionality using free data sources.

### Step 3: Verify Configuration
```bash
# Test that the environment is properly configured
poetry run python -c "import os; print('Environment loaded successfully')"
```

## 🧪 Testing the Installation

### Test 1: Basic System Check
```bash
# Test Python imports
poetry run python -c "
print('Testing imports...')
from src.tools.enhanced_api import get_prices
from src.strategies.strategy_manager import get_strategy_manager
print('✅ All imports successful!')
"
```

### Test 2: Data Provider Test
```bash
# Test data fetching
poetry run python -c "
from src.tools.enhanced_api import get_prices
prices = get_prices('RELIANCE.NS', '2024-01-01', '2024-01-31')
print(f'✅ Fetched {len(prices)} price records for RELIANCE.NS')
"
```

### Test 3: Strategy Framework Test
```bash
# Test strategy manager
poetry run python -c "
from src.strategies.strategy_manager import get_strategy_summary
summary = get_strategy_summary()
print(f'✅ Strategy framework loaded: {summary[\"total_strategies\"]} strategies available')
"
```

### Test 4: NSEUtility Test
```bash
# Test NSEUtility integration
poetry run python -c "
try:
    from src.nsedata.NseUtility import NseUtils
    nse = NseUtils()
    info = nse.get_quote('RELIANCE')
    print(f'✅ NSEUtility working: RELIANCE price ₹{info[\"lastPrice\"]}')
except Exception as e:
    print(f'⚠️ NSEUtility test failed: {e}')
"
```

## 🚀 Running the System

### Quick Start - Indian Stock Analysis
```bash
# Analyze a single Indian stock
poetry run python src/main.py --ticker RELIANCE.NS

# Analyze multiple stocks
poetry run python src/main.py --ticker "RELIANCE.NS,TCS.NS,HDFCBANK.NS"

# Run with detailed reasoning
poetry run python src/main.py --ticker RELIANCE.NS --show-reasoning
```

### Advanced Usage
```bash
# Run with specific date range
poetry run python src/main.py --ticker RELIANCE.NS --start-date 2024-01-01 --end-date 2024-12-31

# Run backtesting
poetry run python src/backtester.py --ticker RELIANCE.NS

# Run with local LLM (if you have Ollama installed)
poetry run python src/main.py --ticker RELIANCE.NS --ollama
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Poetry not found
```bash
# Windows: Add to PATH manually
$env:Path += ";C:\Users\$env:USERNAME\AppData\Roaming\Python\Scripts"

# macOS/Linux: Add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

#### Issue 2: Import errors
```bash
# Make sure you're in the virtual environment
poetry shell

# Reinstall dependencies
poetry install --sync
```

#### Issue 3: Data fetch failures
```bash
# Check internet connection
ping google.com

# Test with a simple request
poetry run python -c "
import requests
response = requests.get('https://www.google.com')
print(f'Internet connection: {response.status_code}')
"
```

#### Issue 4: NSEUtility errors
```bash
# NSEUtility requires specific dependencies
poetry run pip install requests-html beautifulsoup4

# Test NSEUtility separately
poetry run python -c "
from src.nsedata.NseUtility import NseUtils
nse = NseUtils()
print('NSEUtility initialized successfully')
"
```

### Performance Optimization

#### Enable Caching
```bash
# Add to .env file
CACHE_TTL=600  # 10 minutes cache
ENABLE_CACHE=true
```

#### Reduce API Calls
```bash
# Use local data when possible
poetry run python src/main.py --ticker RELIANCE.NS --use-cache
```

## 📊 System Verification

### Complete System Test
```bash
# Run comprehensive system test
poetry run python -c "
print('🧪 Running Complete System Test...')

# Test 1: Data Providers
from src.tools.enhanced_api import get_prices
prices = get_prices('RELIANCE.NS', '2024-01-01', '2024-01-31')
print(f'✅ Data Providers: {len(prices)} records fetched')

# Test 2: Strategy Framework
from src.strategies.strategy_manager import get_strategy_summary
summary = get_strategy_summary()
print(f'✅ Strategy Framework: {summary[\"total_strategies\"]} strategies loaded')

# Test 3: AI Agents
from src.agents.warren_buffett import WarrenBuffettAgent
agent = WarrenBuffettAgent()
print(f'✅ AI Agents: {agent.name} agent loaded')

# Test 4: NSEUtility
try:
    from src.nsedata.NseUtility import NseUtils
    nse = NseUtils()
    info = nse.get_quote('RELIANCE')
    print(f'✅ NSEUtility: RELIANCE price ₹{info[\"lastPrice\"]}')
except Exception as e:
    print(f'⚠️ NSEUtility: {e}')

print('🎉 System test completed!')
"
```

## 🎯 Next Steps

After successful installation:

1. **Read the Documentation**:
   - [Phase 1: Indian Stocks Integration](INDIAN_STOCKS_INTEGRATION.md)
   - [Phase 4: NSEUtility Integration](PHASE4_NSEUTILITY_INTEGRATION.md)

2. **Try Different Features**:
   - Test with different Indian stocks
   - Experiment with strategy framework
   - Explore the enhanced API

3. **Customize the System**:
   - Add your own strategies
   - Configure custom data providers
   - Modify AI agent parameters

## 📞 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the console
2. **Verify prerequisites**: Ensure Python, Git, and Poetry are properly installed
3. **Test components**: Run individual component tests
4. **Check documentation**: Review the phase documentation files
5. **Open an issue**: Create an issue on GitHub with detailed error information

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- The system works without API keys for basic functionality
- All sensitive data is properly excluded via `.gitignore`

---

**Congratulations!** 🎉 You've successfully set up the AI Hedge Fund with Indian Market Integration. You can now start analyzing Indian stocks and running trading strategies! 