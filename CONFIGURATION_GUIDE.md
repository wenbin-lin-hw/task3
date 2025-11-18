# E-Puck Speed World - Configuration Guide

## Overview
This guide provides recommendations for configuring the Genetic Algorithm (GA) and fitness functions for evolving a line-following, obstacle-avoiding robot controller.

---

## 1. Neural Network Architecture

### Recommended Configuration:
```python
self.number_hidden_layer = [8, 6]
```

### Rationale:
- **Two hidden layers** provide sufficient complexity for:
  - Layer 1 (8 neurons): Feature extraction from 11 inputs (8 proximity + 3 ground sensors)
  - Layer 2 (6 neurons): Decision-making layer for motor control
- **Total network**: 11 → 8 → 6 → 2
- **Total weights**: ~150 parameters (manageable for GA optimization)

### Alternative Architectures:
- **Simple (faster evolution)**: `[6]` - Single layer with 6 neurons (~100 weights)
- **Complex (better performance)**: `[10, 8]` - More neurons (~200 weights)

---

## 2. Genetic Algorithm Parameters

### Recommended Configuration:

```python
self.num_generations = 50      # Number of generations
self.num_population = 20       # Population size
self.num_elite = 4             # Elite individuals (20% of population)
```

### Rationale:

#### **num_generations = 50**
- **50 generations** typically sufficient for convergence
- Each generation takes ~150s × 20 individuals = 50 minutes
- Total training time: ~40-50 hours
- Monitor fitness plot; if not converging, increase to 75-100

#### **num_population = 20**
- **20 individuals** balances exploration vs. computation time
- Larger populations (30-40) improve diversity but increase training time
- Smaller populations (10-15) risk premature convergence

#### **num_elite = 4**
- **4 elite individuals** (20% of population) preserved each generation
- Ensures best solutions aren't lost
- Standard practice: 10-30% of population

### Alternative Configurations:

#### Fast Prototyping (Quick Testing):
```python
num_generations = 20
num_population = 10
num_elite = 2
```
- Total time: ~8-10 hours
- Use for testing fitness functions

#### High-Performance (Best Results):
```python
num_generations = 100
num_population = 30
num_elite = 6
```
- Total time: ~125 hours
- Use for final optimization

---

## 3. Fitness Function Design

### Key Principles:
1. **Forward Movement**: Reward speed and forward motion
2. **Line Following**: Reward staying on the line (center sensor detection)
3. **Collision Avoidance**: Penalize proximity sensor activation
4. **Stability**: Penalize spinning and erratic behavior

### Recommended Fitness Functions:

#### **A. Forward Fitness (Speed & Direction)**
```python
# Reward forward movement, penalize backward movement
forwardFitness = (self.velocity_left + self.velocity_right) / 2.0
```

**Explanation:**
- Average wheel velocity encourages forward motion
- Range: -3 to +3 (motors multiplied by 3)
- Negative if moving backward (penalty)

#### **B. Follow Line Fitness (Line Detection)**
```python
# Reward when center sensor detects the line
# Normalized ground sensor values: 0 (white/no line) to 1 (black/line)
followLineFitness = center * 2.0  # Weight: 2x importance
```

**Explanation:**
- `center` is normalized (0-1): 1 when on black line, 0 when off
- Multiply by 2.0 to emphasize line-following importance
- Range: 0 to 2.0

**Alternative (More Sophisticated):**
```python
# Reward center detection, penalize off-center
followLineFitness = center * 2.0 - (left + right) * 0.5
```
- Penalizes when left/right sensors see the line (robot is off-center)

#### **C. Avoid Collision Fitness (Obstacle Avoidance)**
```python
# Penalize proximity sensor activation (normalized 0-1, 1=close obstacle)
# Focus on front sensors (ps0, ps1, ps6, ps7)
front_obstacle = (self.inputs[3] + self.inputs[4] + self.inputs[9] + self.inputs[10]) / 4.0
avoidCollisionFitness = -front_obstacle * 3.0  # Heavy penalty
```

**Explanation:**
- `self.inputs[3:11]` are proximity sensors (ps0-ps7)
- Front sensors: ps0 (inputs[3]), ps1 (inputs[4]), ps6 (inputs[9]), ps7 (inputs[10])
- Average front sensor activation
- Multiply by -3.0 for strong penalty
- Range: -3.0 to 0

**Alternative (All Sensors):**
```python
# Consider all proximity sensors
all_obstacles = sum(self.inputs[3:11]) / 8.0
avoidCollisionFitness = -all_obstacles * 2.0
```

#### **D. Spinning Fitness (Stability)**
```python
# Penalize differential wheel speeds (spinning behavior)
spinningFitness = -abs(self.velocity_left - self.velocity_right) * 0.5
```

**Explanation:**
- Large difference in wheel velocities = spinning/turning sharply
- Absolute difference penalized
- Multiply by -0.5 for moderate penalty
- Range: -3.0 to 0

**Alternative (Encourage Smooth Turns):**
```python
# Allow turning but penalize excessive spinning
speed_diff = abs(self.velocity_left - self.velocity_right)
if speed_diff > 2.0:  # Threshold for "spinning"
    spinningFitness = -(speed_diff - 2.0) * 1.0
else:
    spinningFitness = 0
```

---

## 4. Combined Fitness Function

### Recommended Combination:
```python
combinedFitness = (
    forwardFitness * 1.0 +          # Weight: 1.0 (base speed)
    followLineFitness * 2.0 +       # Weight: 2.0 (most important)
    avoidCollisionFitness * 1.5 +   # Weight: 1.5 (important)
    spinningFitness * 0.5           # Weight: 0.5 (stability)
)
```

### Fitness Component Weights:
- **followLineFitness**: Highest weight (2.0) - primary objective
- **avoidCollisionFitness**: High weight (1.5) - safety critical
- **forwardFitness**: Medium weight (1.0) - speed matters
- **spinningFitness**: Low weight (0.5) - fine-tuning

### Expected Fitness Range:
- **Poor performance**: -5 to 0
- **Moderate performance**: 0 to 3
- **Good performance**: 3 to 6
- **Excellent performance**: 6+

---

## 5. Crossover and Mutation Rates

### Current Settings (in ga.py):
```python
cp = 50  # 50% crossover rate
mp = 30  # 30% mutation rate
```

### Recommendations:

#### **Crossover Rate (cp)**
- **Current: 50%** - Good starting point
- **Alternatives:**
  - 60-70%: More exploitation (use when close to solution)
  - 30-40%: More exploration (use early in evolution)

#### **Mutation Rate (mp)**
- **Current: 30%** - Good for exploration
- **Alternatives:**
  - 20-25%: Less aggressive (use when converging)
  - 40-50%: More aggressive (use if stuck in local optima)

### Adaptive Strategy:
```python
# Early generations: High mutation (exploration)
if generation < num_generations * 0.3:
    mp = 40
# Middle generations: Moderate mutation
elif generation < num_generations * 0.7:
    mp = 30
# Late generations: Low mutation (exploitation)
else:
    mp = 20
```

---

## 6. Training Strategy

### Phase 1: Quick Validation (2-3 hours)
```python
num_generations = 10
num_population = 10
num_elite = 2
```
- **Goal**: Test if fitness function is working
- **Check**: Fitness should increase over generations

### Phase 2: Medium Training (20-30 hours)
```python
num_generations = 50
num_population = 20
num_elite = 4
```
- **Goal**: Get a working controller
- **Check**: Robot should follow line and avoid obstacles

### Phase 3: Fine-Tuning (50-100 hours)
```python
num_generations = 100
num_population = 30
num_elite = 6
```
- **Goal**: Optimize for speed and robustness
- **Check**: Consistent performance over 3 runs

---

## 7. Debugging Tips

### If Fitness Doesn't Improve:
1. **Check sensor normalization**: Ensure min_gs, max_gs, min_ds, max_ds are correct
2. **Reduce fitness complexity**: Start with only forwardFitness + followLineFitness
3. **Increase mutation rate**: Try mp = 50
4. **Check sensor readings**: Print sensor values to verify they're reasonable

### If Robot Spins in Place:
1. **Increase spinningFitness penalty**: Multiply by -1.0 instead of -0.5
2. **Reward forward motion more**: Increase forwardFitness weight

### If Robot Ignores Obstacles:
1. **Increase avoidCollisionFitness penalty**: Multiply by -5.0
2. **Check proximity sensor indices**: Verify inputs[3:11] are correct

### If Robot Doesn't Follow Line:
1. **Increase followLineFitness weight**: Multiply by 3.0 or 4.0
2. **Check ground sensor values**: Print left, center, right values
3. **Adjust min_gs, max_gs**: Ensure line is detected (high values on black)

---

## 8. Expected Results

### Generation 0-10:
- Random behavior, low fitness (0-2)
- Robot may spin, crash, or wander

### Generation 10-30:
- Basic line following emerges (fitness 2-4)
- Some obstacle avoidance behavior

### Generation 30-50:
- Consistent line following (fitness 4-6)
- Good obstacle avoidance
- Reasonable speed

### Generation 50+:
- Optimized performance (fitness 6+)
- Fast, smooth line following
- Reliable obstacle navigation

---

## 9. Final Recommendations

### Best Overall Configuration:
```python
# Neural Network
self.number_hidden_layer = [8, 6]

# GA Parameters
self.num_generations = 50
self.num_population = 20
self.num_elite = 4

# Crossover and Mutation
cp = 50  # 50% crossover
mp = 30  # 30% mutation

# Fitness Functions
forwardFitness = (self.velocity_left + self.velocity_right) / 2.0
followLineFitness = center * 2.0
front_obstacle = (self.inputs[3] + self.inputs[4] + self.inputs[9] + self.inputs[10]) / 4.0
avoidCollisionFitness = -front_obstacle * 3.0
spinningFitness = -abs(self.velocity_left - self.velocity_right) * 0.5

combinedFitness = (
    forwardFitness * 1.0 + 
    followLineFitness * 2.0 + 
    avoidCollisionFitness * 1.5 + 
    spinningFitness * 0.5
)
```

### Training Time Estimate:
- **Per individual**: 150 seconds
- **Per generation**: 150s × 20 = 50 minutes
- **Total (50 generations)**: ~42 hours

---

## 10. Advanced Optimizations (Optional)

### A. Dynamic Fitness Weights
Adjust weights based on generation:
```python
if generation < 20:
    # Early: Focus on line following
    followLineFitness_weight = 3.0
else:
    # Later: Balance speed and line following
    followLineFitness_weight = 2.0
```

### B. Multi-Objective Fitness
Track separate objectives:
```python
# Minimize time (speed)
time_fitness = 1.0 / elapsed_time
# Maximize line coverage
line_fitness = time_on_line / total_time
# Combined
combinedFitness = time_fitness + line_fitness * 2.0
```

### C. Fitness Shaping
Use non-linear transformations:
```python
# Exponential reward for high line detection
followLineFitness = (center ** 2) * 2.0
# Exponential penalty for obstacles
avoidCollisionFitness = -(front_obstacle ** 2) * 3.0
```

---

## Summary Table

| Parameter | Recommended | Fast Test | High Performance |
|-----------|-------------|-----------|------------------|
| **Hidden Layers** | [8, 6] | [6] | [10, 8] |
| **Generations** | 50 | 20 | 100 |
| **Population** | 20 | 10 | 30 |
| **Elite** | 4 | 2 | 6 |
| **Crossover Rate** | 50% | 50% | 60% |
| **Mutation Rate** | 30% | 40% | 25% |
| **Training Time** | ~42 hrs | ~8 hrs | ~125 hrs |

---

Good luck with your evolutionary robotics project! 🤖🧬
