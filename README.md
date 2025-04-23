# MCTF2025 - Maritime Capture The Flag Competition @ AAMAS

A sophisticated simulation environment for developing and testing autonomous strategies in maritime capture-the-flag scenarios. This project provides the official training framework (Pyquaticus) for the 2025 Maritime Capture the Flag Competition at AAMAS.

For competition details, rules, and submission guidelines, visit [mctf2025.com](https://mctf2025.com).

## Competition Overview

The 2025 Maritime Capture the Flag Competition @ AAMAS features:

### Game Rules
- Playing field: 160m x 80m rectangular field divided into two halves
- Teams: Two teams (red and blue) with 3 players each
- Home bases: 10m diameter circular areas containing team flags
- Game duration: 10 minutes per match
- Out-of-bounds results in automatic return to home base

### Scoring System
- Evaluation against top ten ranked teams in round-robin format
- Score = Sum of (Flag captures - Opponent flag captures) across games
- Tiebreaker: Number of flag grabs
- Safety score based on collision counts

### Game Events
- **Tag**: Within 10m of opponent in own half
- **Flag Grab**: Enter opponent's unguarded base and collect flag
- **Flag Capture**: Return opponent's flag to home base without being tagged
- **Auto-Tag**: Triggered when leaving field boundaries
- **Untagging**: Return to home base to remove tagged state

### Eligibility Requirements
- Must be 18 years or older
- Comply with OFAC sanctions regulations
- Not listed on U.S. Statutory Debarment List
- Federal employees/entities may participate
- Teams must be mutually exclusive
- One submission per team allowed

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

1. Clone the repository:
```bash
git clone https://github.com/mit-ll-trusted-autonomy/pyquaticus.git
cd pyquaticus
```

2. Set up a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Agents

### Multi-Agent Reinforcement Learning with RLlib

The project uses RLlib for multi-agent reinforcement learning. Training scripts are provided for both 2v2 and 3v3 scenarios:

```bash
# For 2v2 training
python rl_test/train_2v2.py

# For 3v3 training
python rl_test/train_3v3.py
```

Training checkpoints will be saved in `ray_tests/<checkpoint_num>/policies/<policy-name>`. The checkpoint frequency can be configured in the training scripts.

### Reward Function Design

The environment provides a flexible reward system for training agents. Example reward functions are available in `rewards.py`, including:

- Sparse rewards
- Capture and grab rewards (`caps_and_grabs`)
- Custom reward functions can be implemented with access to:
  - Agent positions and movements
  - Flag status and positions
  - Team scores and statistics
  - Tagging and cooldown states
  - Environment boundaries
  - Collision information
  - LIDAR observations (if enabled)

### Policy Mapping

Agents are mapped to specific policies during training through a policy mapping function. This ensures each agent uses the correct learned policy or opponent policy during training. Example from `train_3v3.py`:

```python
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == 'agent_0':
        return "agent-0-policy"
    if agent_id == 'agent_1':
        return "agent-1-policy"
    if agent_id == 'agent_2':
        return "agent-2-policy"
    return "random"  # Default policy for other agents
```

### Testing and Deployment

The project includes scripts for testing and deploying trained policies:

```bash
# Deploy trained 2v2 policies
python rl_test/deploy_2v2.py path/to/policy1 path/to/policy2

# Deploy trained 3v3 policies
python rl_test/deploy_3v3.py path/to/policy1 path/to/policy2 path/to/policy3
```

Additional testing tools:
- Manual control testing: `test/arrowkeys_test.py`
- Dynamics testing: `test/dynamics_test.py`
- GPS environment testing: `test/gps_env_test.py`
- Random environment testing: `test/rand_env_test.py`

## Competition Submission

### Zip File Submission (Recommended for New Competitors)
1. Navigate to `pyquaticus/rl_test/competition_info`
2. Update observation space settings in `gen_config.py`
3. Implement solution in `solution.py`:
   - `__init__` method to load trained policies
   - `compute_action` method for agent actions
4. Create zip without nested folders:
```bash
zip example.zip -r policies_folder gen_config.py solution.py
```

### Docker Submission
1. Pull base container: `docker pull jkliem/mctf2025:latest`
2. Size limit: 0.25GB over base image size
3. Place solution files in `working_dir` or implement custom communication
4. Push to public DockerHub repository
5. Submit with DockerHub details (username/name:tag)

### Important Notes
- Top scoring submissions must provide algorithm explanation for joint paper publication
- Submissions must run on both RED and BLUE sides
- Multiple accounts or submissions will result in disqualification
- Competition organizers reserve right to disqualify non-compliant entries

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

## Results and Recognition

- Results displayed on MCTF website leaderboard
- Winners announced at AAMAS conference
- Top performers receive unofficial certificates
- No monetary or non-monetary prizes awarded
- Top submissions contribute to joint research paper

Note: This competition is organized by U.S. Naval Research Laboratory, United States Military Academy, and MIT Lincoln Labs, but participation does not create any procurement obligations.

# MCTF2025 - Capture The Flag RL Training

This repository contains a hierarchical reinforcement learning implementation for playing capture-the-flag games.

## Project Structure

```
hrl/
├── checkpoints/     # Saved model checkpoints 
├── environment/     # Game environment implementation
├── policies/        # Hierarchical policy implementation
├── tests/           # Test scripts
├── training/        # Training scripts
├── utils/           # Utility functions and classes
└── visualization/   # Visualization tools
```

## Quick Start

### Training in the Terminal

To train the agent with default settings:

```bash
python hrl/training/train.py
```

For quieter output with minimal logging:

```bash
python train_quiet.py --episodes 5000
```

### Training Parameters

The training script supports several command-line arguments:

- `--num-episodes`: Number of episodes to train (default: 5000)
- `--max-steps`: Maximum steps per episode (default: 500)
- `--num-agents`: Number of agents per team (default: 3)
- `--debug-level`: Debug output level (0=minimal, 1=normal, 2=verbose)
- `--log-dir`: Directory to save logs (default: logs)
- `--checkpoint-dir`: Directory to save model checkpoints (default: hrl/checkpoints)
- `--render`: Enable rendering during training
- `--load-model`: Path to a model to load for continued training

Example with parameters:

```bash
python hrl/training/train.py --num-episodes 10000 --debug-level 1 --log-interval 20
```

### Continuing Training

To continue training from a saved checkpoint:

```bash
python hrl/training/train.py --load-model final_model
```

Or with the quiet script:

```bash
python train_quiet.py --episodes 1000 --load-model final_model
```

## Visualizing Results

After training, visualize the results with:

```bash
python hrl/visualization/plot_metrics.py --log-dir logs --output-dir plots
```

This will create plots showing rewards, win rates, and other metrics over time.

## Testing the Environment

To test the game environment:

```bash
python hrl/tests/test_env.py
```

## Directory Structure

- `hrl/checkpoints/`: Saved model checkpoints from training runs
- `logs/`: Training logs and metrics data
- `plots/`: Generated visualization plots

## Model Files

Trained models are saved with the `.pth` extension in the `hrl/checkpoints/` directory.
By default, the final model is saved as `final_model.pth`.

## Testing Trained Models

You can test a trained model with the `run_test.py` script:

```bash
python run_test.py --model final_model --episodes 5 --render
```

Command line arguments:

- `--model`: Name of the model to load (default: final_model)
- `--checkpoint-dir`: Directory containing model checkpoints (default: hrl/checkpoints)
- `--episodes`: Number of episodes to run (default: 5)
- `--render`: Enable rendering of the environment
- `--debug-level`: Debug output level (0=minimal, 1=normal, 2=verbose)

