# MCTF2025 - Maritime Capture The Flag Simulation Environment

A sophisticated simulation environment for developing and testing autonomous strategies in capture-the-flag scenarios. This project is based on the Moos-IvP-Aquaticus framework and provides a rich environment for both research and training purposes.

## Features

### Game Mechanics
- Team-based capture-the-flag gameplay (Red vs Blue)
- Flag capture and defense mechanics
- Tagging system with cooldowns
- Out-of-bounds detection
- Collision detection
- Comprehensive scoring system (captures, tags, flag grabs)

### Environment
- Support for both GPS-based real-world locations and simulated environments
- Configurable environment bounds and field layout
- Obstacle system (circles and polygons)
- Scrimmage line mechanics
- Topographical features for land and water areas

### Agent Capabilities
- Configurable agent properties (radius, speed, dynamics)
- Multiple dynamics models:
  - Surveyor
  - Heron
  - Large USV
  - Drone
  - Fixed-wing
- LIDAR-based observation system for detecting:
  - Obstacles
  - Flags (team and opponent)
  - Teammates and their status
  - Opponents and their status

### Technical Features
- Reinforcement Learning support
  - Configurable reward functions
  - State and observation history tracking
  - Normalized observation spaces
  - Training configurations for 2v2 and 3v3 scenarios
- Visualization system
  - Configurable rendering options
  - Trajectory visualization
  - LIDAR visualization
  - Video recording capabilities
- Comprehensive testing suite

## Installation

https://mctf2025.com/installation

1. Clone the repository:
https://github.com/mit-ll-trusted-autonomy/pyquaticus/tree/main


## Usage

### Basic Environment Setup
```python
from pyquaticus import PyQuaticus

# Create a basic 2v2 environment
env = PyQuaticus(
    num_agents=4,  # 2v2
    config=get_std_config()
)
```

### Training Agents
```python
# For 2v2 training
python rl_test/train_2v2.py

# For 3v3 training
python rl_test/train_3v3.py
```

### Testing
The project includes various test scripts for different components:
- Random environment testing: `test/rand_env_test.py`
- GPS environment testing: `test/gps_env_test.py`
- Dynamics testing: `test/dynamics_test.py`
- Policy testing: `test/base_policy_test.py`

### Manual Control
The environment supports manual control using arrow keys:
```bash
python test/arrowkeys_test.py
```

## Configuration

The environment is highly configurable through `config.py`. Key configuration options include:
- Game mechanics (scoring, timing, cooldowns)
- Environment setup
- Agent properties
- Observation systems
- Rendering options

## License

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

(C) 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis.

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

SPDX-License-Identifier: BSD-3-Clause 

