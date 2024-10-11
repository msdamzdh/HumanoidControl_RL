# Humanoid Control with Reinforcement Learning

This project implements and compares two different neural network architectures for controlling a humanoid robot in a simulated environment using the REINFORCE algorithm. The two models compared are:
1. A standard Multi-Layer Perceptron (MLP)
2. A Kolmogorov-Arnold Neural Network (KAN)

## Environment

The project uses the `Humanoid-v4` environment from Gymnasium (formerly OpenAI Gym). In this environment, a humanoid robot needs to learn to walk and maintain balance.

## Models

### Common Features
Both implementations share:
- REINFORCE algorithm for policy gradient learning
- Normal distribution for action sampling
- Adam optimizer
- Similar hyperparameters:
  - Learning rate: 1e-3
  - Discount factor (gamma): 0.99

### MLP Model
The MLP model uses standard linear layers:
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, device='cpu', hidden_size1=100, hidden_size2=50):
        self.layer1 = nn.Linear(state_space_dim, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.mean = nn.Linear(hidden_size2, action_space_dim)
        self.log_std = nn.Linear(hidden_size2, action_space_dim)
```

### KAN Model
The KAN model uses Kolmogorov-Arnold Neural Network layers, which are based on the Kolmogorov-Arnold representation theorem. This theorem states that any continuous function of multiple variables can be represented as a composition of one-dimensional functions and additions:
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, device='cpu', hidden_size1=100, hidden_size2=50):
        self.klayer1 = kan.KANLayer(state_space_dim, hidden_size1)
        self.klayer2 = kan.KANLayer(hidden_size1, hidden_size2)
        self.mean = kan.KANLayer(hidden_size2, action_space_dim)
        self.log_std = kan.KANLayer(hidden_size2, action_space_dim)
```

Key differences from MLP:
1. **Theoretical Foundation**: KAN is based on the Kolmogorov-Arnold representation theorem, potentially offering more expressive power for certain types of functions
2. **Architecture**: Uses specialized layers that implement the theorem's principles
3. **Computational Complexity**: Generally more computationally intensive due to its more complex structure

## Results Comparison

| Model | Maximum Reward | Minimum Reward | Training Time | Episodes |
|-------|----------------|----------------|---------------|----------|
| KAN   | 324           | 62             | 10m35s        | 1000     |
| MLP   | 494           | 67             | 1m49s         | 1000     |

### Analysis
1. **Performance**: The MLP model achieved a higher maximum reward (494 vs 324). This suggests that for this particular task, the additional complexity of KAN didn't translate to better performance.
2. **Stability**: Both models had similar minimum rewards (67 vs 62), indicating comparable baseline stability.
3. **Computational Efficiency**: The MLP model trained significantly faster (1m49s vs 10m35s). This aligns with expectations, as KAN layers are more complex and computationally intensive.
4. **Theoretical vs Practical**: Despite KAN's theoretical advantages in function approximation, the simpler MLP architecture performed better in this specific task. This highlights the importance of empirical testing alongside theoretical considerations.

## Implementation Details

### Agent Implementation
Both models use the same REINFORCE agent implementation:
```python
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim, device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, maximize=True)
```

Key features:
- Action sampling using a Normal distribution
- Policy gradient update using collected rewards
- Reward normalization for stable training (in MLP model)

### Training Loop
The training process for both models:
1. Runs for 1000 episodes
2. Records videos every 20-50 episodes
3. Prints progress every 50 episodes
4. Collects total rewards and action counts per episode

## Setup and Running

1. Install required packages:
```bash
pip install gymnasium torch kan
```

2. Run either notebook:
   - `KAN_Model.ipynb` for the KAN implementation
   - `MLP_Model.ipynb` for the MLP implementation

## Conclusions

1. **Architecture Trade-offs**: While KAN offers a theoretically powerful approach based on the Kolmogorov-Arnold representation theorem, the simpler MLP model performed better in this specific task.
2. **Efficiency vs Complexity**: The added complexity of KAN layers resulted in significantly longer training times without performance benefits in this case.
3. **Task Suitability**: The results suggest that for humanoid control, the additional representational power of KAN might not be necessary or beneficial.
4. **Future Work**: 
   - Investigation into tasks where KAN's theoretical advantages might translate to practical benefits
   - Exploration of hybrid architectures that combine KAN and traditional layers
   - Analysis of the impact of different hyperparameters on KAN performance

## Dependencies
- gymnasium
- torch
- kan (for Kolmogorov-Arnold Neural Network implementation)
- matplotlib (for plotting results)
