#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner
Orchestrates all individual test scripts and generates a comprehensive report.
"""

import sys
import os
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from loguru import logger


class ComprehensiveTestRunner:
    """Main test runner for comprehensive testing."""
    
    def __init__(self):
        """Initialize the comprehensive test runner."""
        self.test_results = {
            'data_infrastructure': {},
            'llm_analysis': {},
            'strategies': {},
            'backtesting': {},
            'documentation': {},
            'github_integration': {},
            'summary': {},
            'errors': [],
            'warnings': []
        }
        
        self.test_scripts = {
            'data_infrastructure': 'test_scripts/test_data_infrastructure.py',
            'llm_analysis': 'test_scripts/test_llm_analysis.py',
            'strategies': 'test_scripts/test_strategies.py',
            'backtesting': 'test_scripts/test_backtesting.py'
        }
        
        self.base_dir = os.path.dirname(__file__)
        logger.info("Comprehensive Test Runner initialized")
    
    def run_test_script(self, script_name: str, script_path: str) -> Dict[str, Any]:
        """Run a single test script and capture results."""
        logger.info(f"Running {script_name} test...")
        
        results = {
            'script_name': script_name,
            'script_path': script_path,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': None,
            'exit_code': None,
            'output': '',
            'error': '',
            'status': 'UNKNOWN'
        }
        
        try:
            # Construct full path
            full_path = os.path.join(self.base_dir, script_path)
            
            if not os.path.exists(full_path):
                results['error'] = f"Script not found: {full_path}"
                results['status'] = 'FAIL'
                logger.error(f"❌ Script not found: {full_path}")
                return results
            
            # Run the script
            start_time = time.time()
            
            # Set environment variables for imports
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.join(self.base_dir, '../../..')
            
            process = subprocess.run(
                [sys.executable, full_path],
                capture_output=True,
                text=True,
                cwd=os.path.join(self.base_dir, '../../..'),  # Run from main project directory
                env=env,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            results['end_time'] = datetime.now().isoformat()
            results['duration'] = end_time - start_time
            results['exit_code'] = process.returncode
            results['output'] = process.stdout
            results['error'] = process.stderr
            
            if process.returncode == 0:
                results['status'] = 'PASS'
                logger.info(f"✅ {script_name} test completed successfully in {results['duration']:.2f}s")
            else:
                results['status'] = 'FAIL'
                logger.error(f"❌ {script_name} test failed with exit code {process.returncode}")
            
        except subprocess.TimeoutExpired:
            results['error'] = "Test timed out after 5 minutes"
            results['status'] = 'FAIL'
            logger.error(f"❌ {script_name} test timed out")
        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'FAIL'
            logger.error(f"❌ {script_name} test failed with exception: {e}")
        
        return results
    
    def test_documentation(self) -> Dict[str, Any]:
        """Test documentation completeness."""
        logger.info("Testing documentation...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Check for required documentation files
        required_files = [
            'README.md',
            'USAGE.md',
            'PHASE4_COMPLETION_SUMMARY.md',
            'pyproject.toml'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(self.base_dir, '../../..', file_name)
            if os.path.exists(file_path):
                results['passed'] += 1
                logger.info(f"✅ Documentation file found: {file_name}")
            else:
                results['failed'] += 1
                logger.error(f"❌ Documentation file missing: {file_name}")
            results['total_tests'] += 1
        
        # Check for source code documentation
        src_dirs = [
            'src/agents',
            'src/data',
            'src/strategies',
            'src/ml',
            'src/tools'
        ]
        
        for src_dir in src_dirs:
            dir_path = os.path.join(self.base_dir, '../../..', src_dir)
            if os.path.exists(dir_path):
                # Check for __init__.py files
                init_file = os.path.join(dir_path, '__init__.py')
                if os.path.exists(init_file):
                    results['passed'] += 1
                    logger.info(f"✅ Package structure found: {src_dir}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ Missing __init__.py in: {src_dir}")
                results['total_tests'] += 1
        
        self.test_results['documentation'] = results
        return results
    
    def test_github_integration(self) -> Dict[str, Any]:
        """Test GitHub integration and code check-in status."""
        logger.info("Testing GitHub integration...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        try:
            # Check if git is available
            git_check = subprocess.run(
                ['git', '--version'],
                capture_output=True,
                text=True
            )
            
            if git_check.returncode == 0:
                results['passed'] += 1
                logger.info("✅ Git is available")
            else:
                results['failed'] += 1
                logger.error("❌ Git is not available")
            results['total_tests'] += 1
            
            # Check if we're in a git repository
            git_status = subprocess.run(
                ['git', 'status'],
                capture_output=True,
                text=True,
                cwd=os.path.join(self.base_dir, '../../..')
            )
            
            if git_status.returncode == 0:
                results['passed'] += 1
                logger.info("✅ Git repository found")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Not in a git repository")
            results['total_tests'] += 1
            
            # Check remote origin
            git_remote = subprocess.run(
                ['git', 'remote', '-v'],
                capture_output=True,
                text=True,
                cwd=os.path.join(self.base_dir, '../../..')
            )
            
            if git_remote.returncode == 0 and 'origin' in git_remote.stdout:
                results['passed'] += 1
                logger.info("✅ Git remote origin configured")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Git remote origin not configured")
            results['total_tests'] += 1
            
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ GitHub integration test failed: {e}")
        
        self.test_results['github_integration'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("Starting Comprehensive Test Suite...")
        start_time = time.time()
        
        # Run individual test scripts
        for test_name, script_path in self.test_scripts.items():
            logger.info(f"Running {test_name} test suite...")
            result = self.run_test_script(test_name, script_path)
            self.test_results[test_name] = result
        
        # Run additional tests
        self.test_documentation()
        self.test_github_integration()
        
        # Generate summary
        summary = self.generate_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Comprehensive Test Suite completed in {duration:.2f} seconds")
        logger.info(f"Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        
        return self.test_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_warnings = 0
        
        # Count results from individual test scripts
        for test_name in self.test_scripts.keys():
            result = self.test_results[test_name]
            if result['status'] == 'PASS':
                total_passed += 1
            elif result['status'] == 'FAIL':
                total_failed += 1
            else:
                total_warnings += 1
            total_tests += 1
        
        # Count results from additional tests
        for test_name in ['documentation', 'github_integration']:
            result = self.test_results[test_name]
            total_tests += result.get('total_tests', 0)
            total_passed += result.get('passed', 0)
            total_failed += result.get('failed', 0)
            total_warnings += result.get('warnings', 0)
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'warnings': total_warnings,
            'success_rate': success_rate,
            'status': 'PASS' if success_rate >= 80 else 'FAIL',
            'test_scripts': {
                name: {
                    'status': self.test_results[name]['status'],
                    'duration': self.test_results[name].get('duration', 0)
                }
                for name in self.test_scripts.keys()
            },
            'additional_tests': {
                name: {
                    'tests': self.test_results[name].get('total_tests', 0),
                    'passed': self.test_results[name].get('passed', 0),
                    'success_rate': (self.test_results[name].get('passed', 0) / self.test_results[name].get('total_tests', 1) * 100) if self.test_results[name].get('total_tests', 0) > 0 else 0
                }
                for name in ['documentation', 'github_integration']
            }
        }
        
        self.test_results['summary'] = summary
        return summary
    
    def save_results(self):
        """Save comprehensive test results."""
        results_file = os.path.join(self.base_dir, 'test_results/comprehensive_results.md')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        summary = self.test_results['summary']
        
        with open(results_file, 'w') as f:
            f.write(f"# Comprehensive Test Results\n\n")
            f.write(f"**Timestamp**: {summary['timestamp']}\n")
            f.write(f"**Status**: {summary['status']}\n")
            f.write(f"**Success Rate**: {summary['success_rate']:.1f}%\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total Tests: {summary['total_tests']}\n")
            f.write(f"- Passed: {summary['passed']}\n")
            f.write(f"- Failed: {summary['failed']}\n")
            f.write(f"- Warnings: {summary['warnings']}\n\n")
            
            f.write(f"## Test Script Results\n\n")
            for script_name, script_result in summary['test_scripts'].items():
                f.write(f"### {script_name.replace('_', ' ').title()}\n")
                f.write(f"- Status: {script_result['status']}\n")
                duration = script_result.get('duration', 0) or 0
                f.write(f"- Duration: {duration:.2f}s\n\n")
            
            f.write(f"## Additional Test Results\n\n")
            for test_name, test_result in summary['additional_tests'].items():
                f.write(f"### {test_name.replace('_', ' ').title()}\n")
                f.write(f"- Tests: {test_result['tests']}\n")
                f.write(f"- Passed: {test_result['passed']}\n")
                f.write(f"- Success Rate: {test_result['success_rate']:.1f}%\n\n")
            
            f.write(f"## Detailed Results\n\n")
            f.write(f"```json\n{json.dumps(self.test_results, indent=2)}\n```\n")
        
        print(f"Comprehensive results saved to: {results_file}")


def main():
    """Main test execution function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run comprehensive tests
    runner = ComprehensiveTestRunner()
    results = runner.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*80)
    print("Comprehensive Test Results")
    print("="*80)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Status: {summary['status']}")
    print("="*80)
    
    print("\nTest Script Results:")
    for script_name, script_result in summary['test_scripts'].items():
        status_icon = "✅" if script_result['status'] == 'PASS' else "❌"
        duration = script_result.get('duration', 0) or 0
        print(f"{status_icon} {script_name}: {script_result['status']} ({duration:.2f}s)")
    
    print("\nAdditional Test Results:")
    for test_name, test_result in summary['additional_tests'].items():
        success_rate = test_result['success_rate']
        status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
        print(f"{status_icon} {test_name}: {success_rate:.1f}% ({test_result['passed']}/{test_result['tests']})")
    
    # Save results
    runner.save_results()
    
    return summary['status'] == 'PASS'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 