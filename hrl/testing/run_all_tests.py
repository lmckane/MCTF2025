#!/usr/bin/env python
"""
Main test runner for the HRL testing framework.

This script provides a unified entry point for running tests across the different
test categories in the HRL framework. It can run opponent tests, component tests,
environment tests, or all tests together based on command line arguments.
"""

import argparse
import os
import sys
import importlib.util
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run HRL tests")
    parser.add_argument('--opponent', '-o', action='store_true', help='Run opponent tests')
    parser.add_argument('--component', '-c', action='store_true', help='Run component tests')
    parser.add_argument('--environment', '-e', action='store_true', help='Run environment tests')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests')
    parser.add_argument('--model', '-m', type=str, help='Model path for opponent tests')
    parser.add_argument('--episodes', '-n', type=int, default=5, help='Number of episodes')
    parser.add_argument('--render', '-r', action='store_true', help='Render the environment')
    
    return parser.parse_args()

def import_module(module_path):
    """Import a module from a file path."""
    module_name = os.path.basename(module_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_opponent_tests(args):
    """Run tests against opponents."""
    print("\n" + "="*80)
    print("RUNNING OPPONENT TESTS")
    print("="*80)
    
    # Import the test_against_opponents module
    module_path = Path(__file__).parent / "opponents" / "test_against_opponents.py"
    if not module_path.exists():
        print(f"Error: Could not find {module_path}")
        return
    
    test_module = import_module(str(module_path))
    
    # Create command line arguments for the test
    test_args = ["--episodes", str(args.episodes)]
    if args.model:
        test_args.extend(["--model", args.model])
    if args.render:
        test_args.append("--render")
    
    # Run the test
    original_args = sys.argv
    sys.argv = [str(module_path)] + test_args
    test_module.main()
    sys.argv = original_args

def run_component_tests(args):
    """Run component tests."""
    print("\n" + "="*80)
    print("RUNNING COMPONENT TESTS")
    print("="*80)
    
    # Import the run_tests module
    module_path = Path(__file__).parent / "components" / "run_tests.py"
    if not module_path.exists():
        print(f"Error: Could not find {module_path}")
        return
    
    test_module = import_module(str(module_path))
    
    # Create command line arguments for the test
    test_args = ["--episodes", str(args.episodes)]
    if args.render:
        test_args.append("--render")
    
    # Run the test
    original_args = sys.argv
    sys.argv = [str(module_path)] + test_args
    test_module.main()
    sys.argv = original_args

def run_environment_tests(args):
    """Run environment tests."""
    print("\n" + "="*80)
    print("RUNNING ENVIRONMENT TESTS")
    print("="*80)
    
    # Import and run test_env module
    env_module_path = Path(__file__).parent / "environment" / "test_env.py"
    if env_module_path.exists():
        env_module = import_module(str(env_module_path))
        print("\nRunning environment basic tests...")
        env_module.test_environment()
        env_module.test_territory_checking()
        if args.render:
            env_module.visualize_territories()
    
    # Import and run test_team_coordination module
    team_module_path = Path(__file__).parent / "environment" / "test_team_coordination.py"
    if team_module_path.exists():
        team_module = import_module(str(team_module_path))
        print("\nRunning team coordination tests...")
        team_module.test_team_coordination()

def main():
    """Main entry point for running tests."""
    args = parse_args()
    
    # If no specific tests specified, run all tests
    if not (args.opponent or args.component or args.environment or args.all):
        args.all = True
    
    if args.all or args.opponent:
        run_opponent_tests(args)
    
    if args.all or args.component:
        run_component_tests(args)
    
    if args.all or args.environment:
        run_environment_tests(args)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 