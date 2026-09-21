"""
Test suite for Steps 1 & 2 (located in nlp/ folder as requested in Section 10).
"""
import os
import sys

# Add parent directory to path so root modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_steps_1_2 import run_tests

if __name__ == "__main__":
    run_tests()
