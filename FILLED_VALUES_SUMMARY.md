# Filled Values Summary - E-Puck Speed World Controller

## Quick Reference - All Filled Parameters

### 1. Neural Network Architecture (epuck_starter.py)
```python
self.number_hidden_layer = [8, 6]
```
- **Layer 1**: 8 neurons (feature extraction from sensors)
- **Layer 2**: 6 neurons (decision-making for motor control)
- **Total Architecture**: 11 inputs → 8 hidden → 6 hidden → 2 outputs
- **Total Weights**: ~152 parameters

---

### 2. GA Parameters (supervisorGA_starter.py)
```python
self.num_generations = 50
self.num_population = 20
self.num_elite = 4
```

**Training Time Estimate:**
- Per individual: 150 seconds
- Per generation: 150s × 20 = 50 minutes
- Total: 50 generations × 50 min = ~42 hours

---

### 3. Fitness Functions (epuck_starter.py - calculate_fitness method)

#### A. Forward Fitness
```python
forwardFitness = (self.velocity_left + self.velocity_right) / 2.0
```
- **Purpose**: Reward forward movement and speed
- **Range**: -3.0 to +3.0
- **Explanation**: Average of both wheel velocities. Positive = forward, negative = backward

#### B. Follow Line Fitness
```python
center = self.inputs[1]  # Center ground sensor (0=white, 1=black line)
followLineFitness = center * 2.0
```
- **Purpose**: Reward staying on the black line
- **Range**: 0 to 2.0
- **Weight**: 2.0x (highest priority)
- **Explanation**: Center sensor normalized value multiplied by 2 for emphasis

#### C. Avoid Collision Fitness
```python
front_obstacle = (self.inputs[3] + self.inputs[4] + self.inputs[9] + self.inputs[10]) / 4.0
avoidCollisionFitness = -front_obstacle * 3.0
```
- **Purpose**: Penalize proximity to obstacles
- **Range**: -3.0 to 0
- **Weight**: 3.0x penalty
- **Sensors Used**: 
  - ps0 (inputs[3]) - front-right
  - ps1 (inputs[4]) - front-right-side
  - ps6 (inputs[9]) - front-left-side
  - ps7 (inputs[10]) - front-left
- **Explanation**: Average of front proximity sensors, heavily penalized

#### D. Spinning Fitness
```python
spinningFitness = -abs(self.velocity_left - self.velocity_right) * 0.5
```
- **Purpose**: Penalize spinning/erratic turning behavior
- **Range**: -3.0 to 0
- **Weight**: 0.5x penalty
- **Explanation**: Penalizes large differences in wheel velocities

#### E. Combined Fitness
```python
combinedFitness = (
    forwardFitness * 1.0 + 
    followLineFitness * 2.0 + 
    avoidCollisionFitness * 1.5 + 
    spinningFitness * 0.5
)
```

**Component Weights:**
1. **followLineFitness**: 2.0 (highest - primary objective)
2. **avoidCollisionFitness**: 1.5 (high - safety critical)
3. **forwardFitness**: 1.0 (medium - speed matters)
4. **spinningFitness**: 0.5 (low - fine-tuning)

**Expected Fitness Ranges:**
- Poor: -5 to 0
- Moderate: 0 to 3
- Good: 3 to 6
- Excellent: 6+

---

### 4. Genetic Operators (ga.py)

#### Current Settings:
```python
cp = 50  # 50% crossover rate
mp = 30  # 30% mutation rate
```

**Recommendations:**
- **Keep as is** for standard evolution
- **Increase mp to 40-50%** if evolution stagnates (stuck in local optima)
- **Decrease mp to 20-25%** in later generations for fine-tuning

---

## Sensor Configuration

### Ground Sensors (Normalized 0-1)
```python
min_gs = 0
max_gs = 1000  # Adjusted for typical e-puck ground sensor range
```
- **0**: White surface (no line)
- **1**: Black surface (line detected)
- **Sensors**: left (gs0), center (gs1), right (gs2)

### Proximity Sensors (Normalized 0-1)
```python
min_ds = 0
max_ds = 2400  # Adjusted for typical e-puck proximity sensor range
```
- **0**: No obstacle detected
- **1**: Obstacle very close
- **Sensors**: ps0-ps7 (8 sensors around the robot)

---

## Implementation Steps

### Step 1: Copy Filled Files
Replace your original files with the filled versions:

```bash
# Backup originals
cp epuck_starter/epuck_starter.py epuck_starter/epuck_starter_ORIGINAL.py
cp supervisorGA_starter/supervisorGA_starter.py supervisorGA_starter/supervisorGA_starter_ORIGINAL.py

# Copy filled versions
cp epuck_starter/epuck_starter_FILLED.py epuck_starter/epuck_starter.py
cp supervisorGA_starter/supervisorGA_starter_FILLED.py supervisorGA_starter/supervisorGA_starter.py
```

### Step 2: Quick Test (2-3 hours)
For initial testing, temporarily modify parameters:
```python
self.num_generations = 10
self.num_population = 10
self.num_elite = 2
```

### Step 3: Full Training (42 hours)
Use the recommended parameters:
```python
self.num_generations = 50
self.num_population = 20
self.num_elite = 4
```

### Step 4: Monitor Progress
Watch the fitness plot in Webots:
- **Red line**: Best individual fitness
- **Green line**: Average population fitness
- Both should increase over generations

---

## Troubleshooting Guide

### Problem: Fitness not improving
**Solutions:**
1. Increase mutation rate: `mp = 40`
2. Check sensor values (print statements)
3. Simplify fitness: Start with only `forwardFitness + followLineFitness`
4. Verify sensor normalization ranges

### Problem: Robot spins in place
**Solutions:**
1. Increase spinning penalty: `spinningFitness = -abs(...) * 1.0`
2. Increase forward reward weight: `forwardFitness * 2.0`
3. Add minimum speed requirement to fitness

### Problem: Robot ignores obstacles
**Solutions:**
1. Increase collision penalty: `avoidCollisionFitness = -front_obstacle * 5.0`
2. Verify proximity sensor indices are correct
3. Check max_ds value (should be 2400 for e-puck)

### Problem: Robot doesn't follow line
**Solutions:**
1. Increase line following weight: `followLineFitness = center * 3.0`
2. Check ground sensor values (print left, center, right)
3. Verify max_gs value (should be 1000 for e-puck)
4. Add penalty for off-center: `- (left + right) * 0.5`

---

## Alternative Configurations

### Fast Prototyping (8-10 hours)
```python
# Neural Network
self.number_hidden_layer = [6]

# GA Parameters
self.num_generations = 20
self.num_population = 10
self.num_elite = 2

# Genetic Operators
cp = 50
mp = 40
```

### High Performance (125 hours)
```python
# Neural Network
self.number_hidden_layer = [10, 8]

# GA Parameters
self.num_generations = 100
self.num_population = 30
self.num_elite = 6

# Genetic Operators
cp = 60
mp = 25
```

### Conservative (Slow but Stable)
```python
# Neural Network
self.number_hidden_layer = [8, 6]

# GA Parameters
self.num_generations = 75
self.num_population = 25
self.num_elite = 5

# Genetic Operators
cp = 70
mp = 20
```

---

## Advanced Fitness Functions (Optional Enhancements)

### 1. Enhanced Line Following
```python
# Reward center, penalize sides
followLineFitness = center * 2.0 - (left + right) * 0.5
```

### 2. Speed-Based Reward
```python
# Exponential reward for high speeds
forwardFitness = ((self.velocity_left + self.velocity_right) / 2.0) ** 1.5
```

### 3. Selective Collision Avoidance
```python
# Only penalize when obstacle is very close
if front_obstacle > 0.5:
    avoidCollisionFitness = -(front_obstacle - 0.5) * 5.0
else:
    avoidCollisionFitness = 0
```

### 4. Smooth Turning Reward
```python
# Allow turning but penalize excessive spinning
speed_diff = abs(self.velocity_left - self.velocity_right)
if speed_diff > 2.0:
    spinningFitness = -(speed_diff - 2.0) * 1.0
else:
    spinningFitness = 0
```

### 5. Time-Based Fitness
```python
# Add time tracking to reward faster completion
self.elapsed_time += self.time_step / 1000.0  # Convert to seconds
time_penalty = -self.elapsed_time * 0.01
combinedFitness += time_penalty
```

---

## Expected Evolution Timeline

### Generation 0-5: Random Exploration
- **Fitness**: -2 to 1
- **Behavior**: Random movement, frequent crashes
- **What's happening**: Population exploring solution space

### Generation 5-15: Basic Behaviors Emerge
- **Fitness**: 1 to 3
- **Behavior**: Some forward movement, occasional line detection
- **What's happening**: Selection pressure favoring forward motion

### Generation 15-30: Line Following Develops
- **Fitness**: 3 to 5
- **Behavior**: Consistent line following, some obstacle avoidance
- **What's happening**: Network learning sensor-motor mappings

### Generation 30-50: Optimization Phase
- **Fitness**: 5 to 7+
- **Behavior**: Fast, smooth line following with good obstacle navigation
- **What's happening**: Fine-tuning of weights for optimal performance

---

## Performance Metrics

### Evaluation Criteria (3 runs):
1. **Time to complete circuit**: Lower is better
2. **Line following accuracy**: % of time on line
3. **Collision count**: Should be 0
4. **Smoothness**: Low variance in wheel velocities

### Target Performance:
- **Circuit completion**: < 120 seconds
- **Line accuracy**: > 90%
- **Collisions**: 0
- **Fitness**: > 6.0

---

## Files Modified

1. **epuck_starter/epuck_starter.py**
   - Line 22: `self.number_hidden_layer = [8, 6]`
   - Lines 138-161: All fitness functions filled

2. **supervisorGA_starter/supervisorGA_starter.py**
   - Line 31: `self.num_generations = 50`
   - Line 32: `self.num_population = 20`
   - Line 33: `self.num_elite = 4`

3. **ga.py** (Optional modifications)
   - Line 4: `cp = 50` (can adjust)
   - Line 28: `mp = 30` (can adjust)

---

## Final Checklist

- [ ] Neural network architecture set: `[8, 6]`
- [ ] GA parameters filled: 50 generations, 20 population, 4 elite
- [ ] All fitness functions implemented
- [ ] Sensor normalization ranges adjusted (gs: 1000, ds: 2400)
- [ ] Files saved and ready to run
- [ ] Backup of original files created
- [ ] Webots world file loaded
- [ ] Ready to press 'S' to start evolution

---

## Contact & Support

If you encounter issues:
1. Check the CONFIGURATION_GUIDE.md for detailed explanations
2. Review the troubleshooting section above
3. Monitor fitness plot for convergence issues
4. Adjust parameters based on observed behavior

**Good luck with your evolutionary robotics project!** 🤖🧬🏁
