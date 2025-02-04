@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Adjusted temperature for transformed velocity
    velocity_temp = 0.15  # Reduced temperature for subtler scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Re-scaling uprightness reward to increase influence
    uprightness_temp = 5.0  # Increased temperature to broaden range
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)
    
    # Improved aggregate total reward function to equalize the impact
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
