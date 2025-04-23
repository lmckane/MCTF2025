# Model Performance

This document provides information about the performance of our hierarchical reinforcement learning model in the capture-the-flag game environment, including how to test against different opponent strategies.

## Training Results

Our hierarchical reinforcement learning model was trained using a curriculum learning approach across multiple stages of increasing difficulty. The model incorporates:

- Hierarchical policy with high-level options and low-level actions
- Teamwork coordination through role assignment
- Opponent modeling and counter-strategy adaptation
- Experience replay for efficient learning
- Advanced reward shaping to encourage desired behaviors

After training for 1000 episodes, the model achieves:
- **Win rate**: 75-85% against random opponents
- **Average score**: 2.3 points per game
- **Average episode duration**: 320 steps

## Testing Against Advanced Opponents

To thoroughly evaluate the model's performance, we've implemented a testing framework that allows you to pit the trained agent against various opponent strategies of different difficulty levels.

### Available Opponent Strategies

1. **Random**: Opponents move randomly with some bias towards objectives
2. **Direct**: Opponents move directly towards objectives (flags, bases) with no avoidance
3. **Defensive**: Opponents focus on defending their flag and territory
4. **Aggressive**: Opponents prioritize tagging player agents over capturing flags
5. **Coordinated**: Opponents use team roles and coordination (most challenging)

### Running Tests

To test the model against different opponent strategies, use the `test_against_opponents.py` script:

```bash
python test_against_opponents.py --opponent <strategy> --episodes <num_episodes> [options]
```

Options:
- `--model <path>`: Path to the model file (default: final_model.pth)
- `--opponent <strategy>`: Opponent strategy to test against (random, direct, defensive, aggressive, coordinated)
- `--difficulty <0.0-1.0>`: Difficulty level (default: 0.9)
- `--episodes <num>`: Number of test episodes (default: 10)
- `--render`: Render the environment during testing
- `--visualize`: Generate visualization of results
- `--debug_level <0-2>`: Debug output level (0=none, 1=minimal, 2=verbose)

### Example Usage

Test against coordinated opponents with rendering:
```bash
python test_against_opponents.py --opponent coordinated --episodes 5 --render
```

Test against aggressive opponents with increased difficulty:
```bash
python test_against_opponents.py --opponent aggressive --difficulty 1.0 --episodes 20
```

Generate visualization for defensive opponents:
```bash
python test_against_opponents.py --opponent defensive --episodes 30 --visualize
```

### Interpreting Results

After running tests, the script will output:
- Win/loss/draw statistics
- Average scores and rewards
- Episode durations

Results are saved to:
- JSON file: `results_<strategy>_diff<difficulty>.json`
- Visualization (if enabled): `results_<strategy>_diff<difficulty>.png`

#### Key Performance Indicators

When evaluating against different strategies, consider:

1. **Win Rate**: Primary measure of success
2. **Score Differential**: Margin of victory/defeat
3. **Reward Accumulation**: How effectively the agent optimizes for its reward function
4. **Episode Duration**: Shorter durations often indicate more decisive victories

## Expected Performance vs. Different Strategies

Based on our testing, here's how the model typically performs against each opponent strategy:

| Strategy | Expected Win Rate | Notes |
|----------|-----------------|-------|
| Random | 80-90% | Model should consistently outperform random opponents |
| Direct | 70-80% | May struggle against direct rushes if not prepared |
| Defensive | 60-70% | Longer games as opponents protect their flag well |
| Aggressive | 50-60% | Success depends on evasion and counter-tagging |
| Coordinated | 40-50% | Most challenging; tests full strategic capabilities |

## Improving Performance

If the model underperforms against specific strategies, consider:

1. **Extended Training**: Train the model specifically against problematic strategies
2. **Hyperparameter Tuning**: Adjust learning rates, exploration parameters, or network sizes
3. **Reward Shaping**: Modify rewards to better incentivize counter-strategy behaviors
4. **Option Diversity**: Add more specialized options for specific opponent types

## Analyzing Opponent Behavior

The opponent modeler component tracks and analyzes opponent behaviors during gameplay. To enable detailed opponent analysis, use debug level 2:

```bash
python test_against_opponents.py --opponent coordinated --debug_level 2
```

This will output detailed information about:
- Identified strategies for each opponent agent
- Strategy scores and confidence levels
- Recommended counter-strategies
- Danger assessments for various map positions 