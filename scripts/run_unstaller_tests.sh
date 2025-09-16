#!/usr/bin/env bash
# Un-Staller Test Harness - CI/Local Runner
# Ensures all tests either pass or fail, never hang
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
INTEGRATION="${INTEGRATION:-0}"
TEST_TIMEOUT="${TEST_TIMEOUT:-90}"
GLOBAL_TIMEOUT="${GLOBAL_TIMEOUT:-1800}"
PYTHON="${PYTHON:-python3}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[Un-Staller]${NC} $*"
}

error() {
    echo -e "${RED}[Un-Staller ERROR]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[Un-Staller]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[Un-Staller WARNING]${NC} $*"
}

# Help function
show_help() {
    cat << EOF
Un-Staller Test Harness - Guarantees all tests pass or fail, never hang

Usage: $0 [OPTIONS]

Options:
    --integration         Enable integration tests (requires live services)
    --timeout SECONDS     Timeout per test file (default: ${TEST_TIMEOUT})
    --global-timeout SEC  Global timeout for all tests (default: ${GLOBAL_TIMEOUT})
    --python PYTHON       Python executable to use (default: ${PYTHON})
    --help               Show this help

Environment Variables:
    INTEGRATION          Set to 1 to enable integration tests
    TEST_TIMEOUT         Timeout per test file in seconds
    GLOBAL_TIMEOUT       Global timeout for all tests in seconds
    PYTHON               Python executable path

Examples:
    # Run with mocked dependencies (default, safe)
    $0

    # Run with live integration tests
    $0 --integration

    # Run with custom timeouts
    $0 --timeout 60 --global-timeout 900

    # In CI (GitHub Actions example)
    INTEGRATION=0 $0
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --integration)
            INTEGRATION=1
            shift
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --global-timeout)
            GLOBAL_TIMEOUT="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate Python
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    error "Python executable not found: $PYTHON"
    exit 1
fi

# Check Python version (require 3.8+)
PYTHON_VERSION=$($PYTHON -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if [[ $(echo "$PYTHON_VERSION 3.8" | tr ' ' '\n' | sort -V | head -n1) != "3.8" ]]; then
    error "Python 3.8+ required, found: $PYTHON_VERSION"
    exit 1
fi

# Verify workspace structure
if [[ ! -f "$WORKSPACE_ROOT/requirements.txt" ]]; then
    error "Invalid workspace: requirements.txt not found in $WORKSPACE_ROOT"
    exit 1
fi

if [[ ! -d "$WORKSPACE_ROOT/tests" ]]; then
    error "Invalid workspace: tests directory not found in $WORKSPACE_ROOT"
    exit 1
fi

# Pre-flight checks
log "Un-Staller Test Harness Starting..."
log "Workspace: $WORKSPACE_ROOT"
log "Python: $PYTHON ($PYTHON_VERSION)"
log "Integration: $([ "$INTEGRATION" = "1" ] && echo "ENABLED" || echo "DISABLED")"
log "Timeouts: ${TEST_TIMEOUT}s per test, ${GLOBAL_TIMEOUT}s global"

# Check if virtual environment is active (recommended)
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ ! -f "$WORKSPACE_ROOT/.venv/bin/activate" ]]; then
    warn "No virtual environment detected. Consider using: python -m venv .venv && source .venv/bin/activate"
fi

# Install/verify dependencies
log "Checking dependencies..."
cd "$WORKSPACE_ROOT"

# Check if pytest is available with required plugins
if ! $PYTHON -c "import pytest, pytest_timeout, pytest_asyncio" >/dev/null 2>&1; then
    log "Installing/updating test dependencies..."
    $PYTHON -m pip install --upgrade -q \
        pytest>=6.0 \
        pytest-timeout>=2.3.1 \
        pytest-xdist>=3.6.1 \
        pytest-asyncio>=0.23.8 \
        pytest-rerunfailures>=13.0 \
        faulthandler
fi

# Validate configuration files
if [[ ! -f "$WORKSPACE_ROOT/pytest.ini" ]]; then
    warn "pytest.ini not found - some timeout settings may not apply"
fi

if [[ ! -f "$WORKSPACE_ROOT/tests/conftest.py" ]]; then
    warn "tests/conftest.py not found - test isolation may be reduced"
fi

# Set environment for the test runner
export INTEGRATION
export TEST_TIMEOUT
export GLOBAL_TIMEOUT
export WORKSPACE_ROOT

# Handle signals gracefully
cleanup() {
    local exit_code=$?
    log "Cleaning up..."
    # Kill any remaining pytest processes
    pkill -f "pytest.*$WORKSPACE_ROOT" 2>/dev/null || true
    exit $exit_code
}
trap cleanup EXIT INT TERM

# Run the Un-Staller test harness
log "Launching Un-Staller test harness..."

HARNESS_ARGS=(
    --workspace "$WORKSPACE_ROOT"
    --timeout "$TEST_TIMEOUT"
    --global-timeout "$GLOBAL_TIMEOUT"
)

if [[ "$INTEGRATION" = "1" ]]; then
    HARNESS_ARGS+=(--integration)
fi

if $PYTHON "$SCRIPT_DIR/run_unstaller_tests.py" "${HARNESS_ARGS[@]}"; then
    success "✅ All tests completed successfully!"
    
    # Look for the most recent results file
    LATEST_RESULTS=$(find "$WORKSPACE_ROOT" -name "test_results_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo "")
    
    if [[ -n "$LATEST_RESULTS" ]] && [[ -f "$LATEST_RESULTS" ]]; then
        log "📊 Latest results: $LATEST_RESULTS"
        
        # Extract summary for quick review
        if command -v jq >/dev/null 2>&1; then
            SUMMARY=$(jq -r '.summary | "Passed: \(.passed), Failed: \(.failed), Timeouts: \(.timeouts), Duration: \(.duration | floor)s"' "$LATEST_RESULTS" 2>/dev/null || echo "Summary unavailable")
            log "📈 $SUMMARY"
        fi
    fi
    
    exit 0
else
    EXIT_CODE=$?
    error "❌ Test harness completed with failures (exit code: $EXIT_CODE)"
    
    # Show brief failure summary if available
    LATEST_RESULTS=$(find "$WORKSPACE_ROOT" -name "test_results_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo "")
    
    if [[ -n "$LATEST_RESULTS" ]] && [[ -f "$LATEST_RESULTS" ]] && command -v jq >/dev/null 2>&1; then
        FAILED_COUNT=$(jq -r '.summary.failed' "$LATEST_RESULTS" 2>/dev/null || echo "0")
        TIMEOUT_COUNT=$(jq -r '.summary.timeouts' "$LATEST_RESULTS" 2>/dev/null || echo "0")
        
        if [[ "$FAILED_COUNT" != "0" ]] || [[ "$TIMEOUT_COUNT" != "0" ]]; then
            error "💥 ${FAILED_COUNT} failures, ${TIMEOUT_COUNT} timeouts detected"
            log "🔍 Check detailed results in: $LATEST_RESULTS"
        fi
    fi
    
    exit $EXIT_CODE
fi
