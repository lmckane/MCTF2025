#!/usr/bin/env python
"""
Backward compatibility script for the HRL testing framework.

This script provides compatibility with previous test paths and entry points.
It redirects old paths to the new consolidated testing structure.
"""

import sys
import os
import importlib
from pathlib import Path

def import_from_new_path(old_path, new_path):
    """
    Import a module from the new path when the old path is requested.
    
    Args:
        old_path: Original module path that might be used in imports
        new_path: New consolidated path where the module is actually located
    """
    sys.modules[old_path] = importlib.import_module(new_path)
    return sys.modules[old_path]

# Dictionary mapping old import paths to new paths
IMPORT_MAP = {
    'hrl.tests.test_opponents': 'hrl.testing.opponents.test_against_opponents',
    'hrl.tests.test_hrl_components': 'hrl.testing.components.test_hrl',
    'hrl.tests.test_environment': 'hrl.testing.environment.test_env',
    'hrl.tests.test_coordination': 'hrl.testing.environment.test_team_coordination',
    'hrl.test.utils': 'hrl.testing.utils.test_utils',
    # Add any other mappings here
}

def setup_backward_compatibility():
    """Setup backward compatibility for all paths in the import map."""
    for old_path, new_path in IMPORT_MAP.items():
        try:
            import_from_new_path(old_path, new_path)
            print(f"Redirected {old_path} to {new_path}")
        except ImportError:
            print(f"Warning: Could not set up redirection from {old_path} to {new_path}")

def run_legacy_script(script_path):
    """
    Run a legacy script by redirecting to its equivalent in the new structure.
    
    Args:
        script_path: Path to the legacy script
    """
    script_name = os.path.basename(script_path)
    
    # Map old script names to new paths
    script_map = {
        'test_opponents.py': 'hrl/testing/opponents/test_against_opponents.py',
        'test_hrl.py': 'hrl/testing/components/test_hrl.py',
        'run_hrl_tests.py': 'hrl/testing/components/run_tests.py',
        'test_env.py': 'hrl/testing/environment/test_env.py',
        'test_team_coordination.py': 'hrl/testing/environment/test_team_coordination.py',
    }
    
    if script_name in script_map:
        new_path = script_map[script_name]
        print(f"Redirecting to new script location: {new_path}")
        # Execute the new script
        new_script_path = str(Path(os.getcwd()) / new_path)
        exec(open(new_script_path).read())
    else:
        print(f"Error: No mapping found for legacy script {script_name}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_legacy_script(sys.argv[1])
    else:
        setup_backward_compatibility()
        print("Backward compatibility setup complete.") 