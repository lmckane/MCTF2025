# HRL Testing Framework

This directory contains the consolidated testing framework for the Hierarchical Reinforcement Learning (HRL) system.

## Directory Structure

- `/opponents/`: Tests for evaluating trained models against different opponent strategies
  - `test_against_opponents.py`: Script for testing models against various opponent types
  
- `/components/`: Tests for HRL components like options and policies
  - `run_tests.py`: Script for running component tests
  - `test_hrl.py`: Core HRL testing utilities
  
- `/environment/`: Tests for environment functionality and coordination
  - `test_env.py`: Tests for the game environment with random actions and territory visualization
  - `test_team_coordination.py`: Tests for the team coordination system with role assignments
  
- `/utils/`: Shared testing utilities and helpers
  - `test_utils.py`: Common testing functions for state creation, visualization, and analysis

## Running Tests

### Testing Against Opponents

Test a trained model against different opponent strategies:

```bash
python -m hrl.testing.opponents.test_against_opponents --model my_model --episodes 10 --render
```

Options:
- `--model`: Model name to load (default: final_model)
- `--episodes`: Number of episodes to run per opponent (default: 5)
- `--render`: Render the environment
- `--opponent`: Specific opponent to test against (default: test all)
- `--difficulty`: Difficulty level from 0.0 to 1.0 (default: 0.8)

### Testing HRL Components

Test the individual components of the HRL system:

```bash
python -m hrl.testing.components.run_tests
```

This will test various options (attack, defend, etc.) and the full HRL system.

### Testing Environment

Test the game environment and team coordination:

```bash
python -m hrl.testing.environment.test_env
python -m hrl.testing.environment.test_team_coordination
```

### Using Test Utilities

The `utils/test_utils.py` module provides common functions for testing:

```python
from hrl.testing.utils.test_utils import create_test_env, create_complex_state, visualize_state

# Create a test environment
env = create_test_env()

# Create a complex state for testing
state = create_complex_state(env)

# Visualize the state
fig, ax = visualize_state(state, "Test Visualization")
```

### Running All Tests

The `run_all_tests.py` script lets you run all tests in the framework:

```bash
# Run all tests
python -m hrl.testing.run_all_tests --all

# Run only component tests
python -m hrl.testing.run_all_tests --components

# Run only environment tests
python -m hrl.testing.run_all_tests --environment

# Run only opponent tests
python -m hrl.testing.run_all_tests --opponents

# Run with specific options
python -m hrl.testing.run_all_tests --model my_model --episodes 3 --render
```

## Migration from Old Test Directories

A migration script is provided to help transition tests from the old structure:

```bash
# See what would be moved without copying anything
python -m hrl.testing.migrate_tests --dry-run

# Actually migrate the files
python -m hrl.testing.migrate_tests
```

This will copy any remaining test files from `hrl/tests/` and `hrl/test/` to their proper locations in the new structure.

## Consolidation Note

This testing framework consolidates previously separate test directories (`hrl/tests/` and `hrl/test/`) into a unified structure for better maintainability. All new tests should be added to this framework. 