#!/usr/bin/env python3
"""
Minimal test file to verify Un-Staller harness functionality
"""
import time
import pytest

def test_passes():
    """Test that should pass quickly"""
    assert True

def test_slow_pass():
    """Test that takes time but should pass"""
    time.sleep(1)
    assert True

@pytest.mark.skip(reason="Intentional failure test - skipped for harness validation")
def test_fails():
    """Test that should fail"""
    assert False, "This test intentionally fails"
