## Goal
Train **multi-agent, hierarchical reinforcement learning (HRL)** policies in **Pyquaticus** that:
1. **Use reward shaping** to guide learning efficiently.
2. **Exploit role switching** and **behavior primitives** to simplify decision-making.
3. Are **more offensive** than previous Pyquaticus baselines.

---

## Architecture

### 1. **Use Hierarchical Reinforcement Learning (HRL)**
Inspired by the [Towards Data Science HRL article](https://towardsdatascience.com/hierarchical-reinforcement-learning-56add31a21ab) and the official Aquaticus paper:

**Top-level policy**: Chooses discrete high-level behaviors ("options"):
- `attack_flag`
- `guard_flag`
- `evade_opponent`
- `tag_intruder`
- `retreat_home`

**Low-level controllers (options)**:
- Each of the above is a separately trained policy (can be PPO or DDQN), or a hand-coded behavior to start with (MOOS-IvP-style).

**Why this helps:**
- Solves the **credit assignment** problem by breaking long sparse reward traces into subgoals.
- Matches the *Options Framework* described on page 6 of the official MCTF competition paper.

---

### 2. **Reward Shaping Strategy**
Use additive shaping terms based on the 2311.16339v1 paper:
- +1 for nearing opponent flag (with diminishing returns)
- -1 for collision proximity
- +0.5 for guarding flag zone if opponent detected
- +2 for successful tag
- -0.5 for loitering without action > N steps

This shapes sparse reward functions into *progressive signals*, essential in multi-agent settings where game events are infrequent.

---

### 3. **Training Plan**
#### Tools:
- [x] **Pyquaticus** (already installed)
- [x] **Ray RLlib** with PettingZoo wrapper
- [ ] Optional: Stable Baselines3 (if you want easier integration with PPO/TD3/DDQN)

#### Baseline Phase:
- Start with **pre-trained low-level options** (e.g., using imitation learning on Pav01 agents or scripted behaviors).
- Implement a **meta-controller** policy that learns which option to call depending on state.

```python
# Pseudocode for HRL meta-agent
if opponent_near_flag:
    select("tag_intruder")
elif teammate_tagged and I have flag:
    select("retreat_home")
elif flag_exposed:
    select("attack_flag")
else:
    select("guard_flag")
```

#### Curriculum Learning (optional but recommended):
- Train in stages:
  1. 1v1 attacker vs defender
  2. Add teammates
  3. Introduce stronger adversaries (Easy ➝ Medium ➝ RL)

---

### 4. **Evaluation Metrics**
Track:
- Mean Grabs per game
- Mean Captures per game
- Agent movement heatmaps
- Win rate vs Pav01 / MOOS-based opponents

Compare with Table II from the Aquaticus report to quantify improvement over the current Pyquaticus baseline.

---