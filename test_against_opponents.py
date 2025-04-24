#!/usr/bin/env python
"""
Wrapper script that redirects to the canonical test script in hrl/testing/opponents/test_against_opponents.py.
This script is provided for convenience to allow running the test from the project root.
"""

import sys
import os
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from pyquaticus import pyquaticus_v0
import pyquaticus.config
from pyquaticus.envs.pyquaticus import Team
import pyquaticus.utils.rewards as rewards

from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor

def main():
    """Load and run the canonical test script."""
    # Get the path to the canonical script
    script_path = os.path.join('hrl', 'testing', 'opponents', 'test_against_opponents.py')
    
    if not os.path.exists(script_path):
        print(f"Error: Canonical test script not found at {script_path}")
        print("Please ensure the project structure is intact.")
        return 1
    
    print(f"Running canonical test script from {script_path}")
    print(f"IMPORTANT: Using the official Pyquaticus environment for testing")
    
    # Import the module from the canonical path
    spec = importlib.util.spec_from_file_location("test_against_opponents", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run the main function from the imported module
    return module.main()

if __name__ == "__main__":
    sys.exit(main()) 