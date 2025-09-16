#!/usr/bin/env python3
"""
Un-Staller Test Harness - Ensures all tests either pass or fail, never hang
Guarantees every test file gets a conclusive PASS/FAIL status with timeout protection.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Test discovery and execution configuration
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "90"))  # seconds per test file
GLOBAL_TIMEOUT = int(os.getenv("GLOBAL_TIMEOUT", "1800"))  # 30 minutes total
ENABLE_INTEGRATION = os.getenv("ENABLE_INTEGRATION", "0") == "1"

def test_harness_configuration():
    """Test that the harness configuration is valid"""
    assert TEST_TIMEOUT > 0, "Test timeout must be positive"
    assert GLOBAL_TIMEOUT > 0, "Global timeout must be positive"
    assert GLOBAL_TIMEOUT > TEST_TIMEOUT, "Global timeout must be greater than test timeout"

class UnStallerTestResult:
    def __init__(self, name: str):
        self.name = name
        self.status: Optional[str] = None  # PASS, FAIL, TIMEOUT, SKIP
        self.duration: float = 0.0
        self.output: str = ""
        self.error: str = ""
        self.traceback: str = ""

class UnStallerHarness:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.results: Dict[str, UnStallerTestResult] = {}
        self.start_time = time.time()
        self.test_timeout = TEST_TIMEOUT
        self.global_timeout = GLOBAL_TIMEOUT
        self.enable_integration = ENABLE_INTEGRATION
        
    def discover_test_files(self) -> List[Path]:
        """Find all test files in the repository (excluding dependencies)"""
        test_files = []
        
        # Look for files with 'test' in the name in specific directories
        # to avoid scanning the entire filesystem
        search_dirs = [
            self.workspace_root,  # Root level
            self.workspace_root / "tests",  # Standard tests directory
            self.workspace_root / "test",   # Alternative tests directory
            self.workspace_root / "scripts", # Scripts that might contain tests
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            # Search in directory and immediate subdirectories only (depth=2)
            for pattern in ["test*.py", "*test*.py", "Test*.py", "*Test*.py"]:
                # Direct files in the directory
                test_files.extend(search_dir.glob(pattern))
                # Files in immediate subdirectories
                for subdir in search_dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        test_files.extend(subdir.glob(pattern))
        
        # Filter out unwanted files
        filtered = []
        for test_file in test_files:
            rel_path = str(test_file.relative_to(self.workspace_root))
            
            # Skip common dependency/build directories
            skip_patterns = [
                "__pycache__", ".venv", "venv", "node_modules", 
                ".git", "site-packages", "lib/python", "source/lib",
                ".pytest_cache", "build", "dist"
            ]
            
            # Skip conftest.py files - they're pytest configuration, not tests
            if test_file.name == "conftest.py":
                continue
            
            if any(pattern in rel_path for pattern in skip_patterns):
                continue
                
            # Only include Python files that exist and are readable
            if test_file.is_file() and test_file.suffix == '.py':
                filtered.append(test_file)
        
        return sorted(set(filtered))

    def setup_test_environment(self):
        """Configure environment for safe test execution"""
        # Force stub usage unless integration is explicitly enabled
        if not self.enable_integration:
            os.environ["USE_NATS_STUBS"] = "1"
            os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        
        # Enable debugging aids
        os.environ["PYTHONASYNCIODEBUG"] = "1"
        os.environ["PYTHONFAULTHANDLER"] = "1"
        
        # Configure pytest-asyncio
        os.environ["PYTEST_ASYNCIO_MODE"] = "auto"
        
        # Dummy keys for libraries that require them
        dummy_keys = {
            "OPENAI_API_KEY": "test_key",
            "ANTHROPIC_API_KEY": "test_key", 
            "HUGGINGFACEHUB_API_TOKEN": "test_key"
        }
        for key, value in dummy_keys.items():
            os.environ.setdefault(key, value)
    
    def run_single_test_file(self, test_file: Path) -> UnStallerTestResult:
        """Run a single test file with timeout protection"""
        result = UnStallerTestResult(str(test_file.relative_to(self.workspace_root)))
        
        start_time = time.time()
        
        # Build pytest command using current Python interpreter
        # Use relative path from workspace root
        relative_path = test_file.relative_to(self.workspace_root)
        cmd = [
            sys.executable, "-m", "pytest",
            str(relative_path),
            "-v", "--tb=short", "--no-header"
        ]
        
        if not self.enable_integration:
            cmd.extend(["-m", "not integration"])
        
        # Set timeout via environment variables
        env = os.environ.copy()
        env["PYTEST_TIMEOUT"] = str(self.test_timeout)
        env["PYTEST_TIMEOUT_METHOD"] = "thread"
        
        # Debug: Print command being executed (simplified)
        print(f"    🔧 Running pytest with {self.test_timeout}s timeout...")
        print(f"    📄 Test file: {relative_path}")
        
        try:
            # Use subprocess with timeout - cmd is built from safe components
            # Change to the workspace root so relative paths work correctly
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workspace_root.absolute()),
                env=env  # Pass environment with timeout settings
            )
            
            try:
                stdout, stderr = proc.communicate(timeout=self.test_timeout + 10)
                result.output = stdout
                result.error = stderr
                
                if proc.returncode == 0:
                    result.status = "PASS"
                else:
                    result.status = "FAIL"
                    
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()  # Clean up
                result.status = "TIMEOUT"
                result.error = f"Test file timed out after {self.test_timeout} seconds"
                
        except Exception as e:
            result.status = "FAIL"
            result.error = f"Failed to execute test: {e}"
            result.traceback = str(e)
        
        result.duration = time.time() - start_time
        return result
    
    def run_all_tests(self) -> bool:
        """Execute all discovered test files"""
        test_files = self.discover_test_files()
        
        if not test_files:
            print("❌ No test files discovered")
            return False
        
        print(f"🔍 Discovered {len(test_files)} test files")
        print(f"⚙️  Integration mode: {'ENABLED' if self.enable_integration else 'DISABLED (mocked)'}")
        print(f"⏱️  Timeout per file: {self.test_timeout}s, Global: {self.global_timeout}s")
        print("=" * 80)
        
        self.setup_test_environment()
        
        # Process each test file
        for i, test_file in enumerate(test_files, 1):
            # Check global timeout
            elapsed = time.time() - self.start_time
            if elapsed > self.global_timeout:
                print(f"\n⏰ Global timeout ({self.global_timeout}s) exceeded, stopping")
                break
            
            remaining_time = self.global_timeout - elapsed
            print(f"\n[{i}/{len(test_files)}] Running {test_file.name} (⏱️ {remaining_time:.0f}s remaining)")
            
            result = self.run_single_test_file(test_file)
            self.results[result.name] = result
            
            # Print immediate result
            status_emoji = {
                "PASS": "✅",
                "FAIL": "❌", 
                "TIMEOUT": "⏰",
                "SKIP": "⏭️"
            }.get(result.status, "❓")
            
            print(f"  {status_emoji} {result.status} ({result.duration:.1f}s)")
            
            if result.status in ["FAIL", "TIMEOUT"] and result.error:
                # Show brief error for failures
                error_lines = result.error.split('\n')[:3]  # First 3 lines
                for line in error_lines:
                    if line.strip():
                        print(f"    💬 {line.strip()}")
        
        return self.generate_summary()
    
    def generate_summary(self) -> bool:
        """Generate final summary and save detailed results"""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.status == "PASS")
        failed = sum(1 for r in self.results.values() if r.status == "FAIL")
        timeouts = sum(1 for r in self.results.values() if r.status == "TIMEOUT")
        skipped = sum(1 for r in self.results.values() if r.status == "SKIP")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        total_duration = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("🎯 UN-STALLER TEST HARNESS SUMMARY")
        print("=" * 80)
        print(f"📊 Results: {passed} PASS, {failed} FAIL, {timeouts} TIMEOUT, {skipped} SKIP")
        print(f"📈 Success Rate: {success_rate:.1f}% ({passed}/{total})")
        print(f"⏱️  Total Duration: {total_duration:.1f}s")
        print(f"🔧 Integration: {'ENABLED' if self.enable_integration else 'DISABLED'}")
        
        # List failures and timeouts
        if failed > 0 or timeouts > 0:
            print("\n❌ FAILED/TIMEOUT TESTS:")
            for name, result in self.results.items():
                if result.status in ["FAIL", "TIMEOUT"]:
                    print(f"  • {name} - {result.status} ({result.duration:.1f}s)")
        
        # Save detailed machine-readable results
        self.save_detailed_results()
        
        # Return True if no failures or timeouts (all pass/skip is OK)
        return failed == 0 and timeouts == 0
    
    def save_detailed_results(self):
        """Save detailed results in multiple formats"""
        timestamp = int(time.time())
        
        # JSON format for machines
        json_results = {
            "timestamp": timestamp,
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results.values() if r.status == "PASS"),
                "failed": sum(1 for r in self.results.values() if r.status == "FAIL"),
                "timeouts": sum(1 for r in self.results.values() if r.status == "TIMEOUT"),
                "skipped": sum(1 for r in self.results.values() if r.status == "SKIP"),
                "duration": time.time() - self.start_time,
                "integration_enabled": self.enable_integration
            },
            "results": {
                name: {
                    "status": r.status,
                    "duration": r.duration,
                    "output": r.output,
                    "error": r.error,
                    "traceback": r.traceback
                }
                for name, r in self.results.items()
            }
        }
        
        json_path = self.workspace_root / f"test_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {json_path}")
        
        # Human-readable summary
        summary_path = self.workspace_root / f"test_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write("Un-Staller Test Harness Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {time.ctime(timestamp)}\n")
            f.write(f"Integration Mode: {'ENABLED' if self.enable_integration else 'DISABLED'}\n")
            f.write(f"Total Tests: {len(self.results)}\n\n")
            
            for status in ["PASS", "FAIL", "TIMEOUT", "SKIP"]:
                tests = [name for name, r in self.results.items() if r.status == status]
                if tests:
                    f.write(f"{status} ({len(tests)}):\n")
                    for test in sorted(tests):
                        duration = self.results[test].duration
                        f.write(f"  • {test} ({duration:.1f}s)\n")
                    f.write("\n")
        
        print(f"📄 Human summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Un-Staller Test Harness")
    parser.add_argument("--integration", action="store_true", 
                       help="Enable integration tests (requires live services)")
    parser.add_argument("--timeout", type=int, default=TEST_TIMEOUT,
                       help=f"Timeout per test file in seconds (default: {TEST_TIMEOUT})")
    parser.add_argument("--global-timeout", type=int, default=GLOBAL_TIMEOUT,
                       help=f"Global timeout for all tests (default: {GLOBAL_TIMEOUT})")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                       help="Workspace root directory")
    
    args = parser.parse_args()
    
    # Update configuration with CLI args
    test_timeout = args.timeout
    global_timeout = args.global_timeout
    enable_integration = args.integration or ENABLE_INTEGRATION
    
    # Override environment variables
    if enable_integration:
        os.environ["ENABLE_INTEGRATION"] = "1"
    os.environ["TEST_TIMEOUT"] = str(test_timeout)
    os.environ["GLOBAL_TIMEOUT"] = str(global_timeout)
    
    harness = UnStallerHarness(args.workspace)
    harness.test_timeout = test_timeout
    harness.global_timeout = global_timeout
    harness.enable_integration = enable_integration
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, generating partial results...")
        harness.generate_summary()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = harness.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        harness.generate_summary()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
