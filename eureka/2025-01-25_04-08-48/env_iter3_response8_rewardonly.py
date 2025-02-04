@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Compute the velocity reward
    velocity_change = potentials - prev_potentials

    # Adjust velocity reward with a temperature parameter
    velocity_temp = 0.3
    transformed_velocity_reward = torch.exp(velocity_change * velocity_temp) - 1.0

    # Redefine stability reward to incorporate smoothness of movement
    position_temp = 0.1
    torsoness_temp = 0.05
    
    # Penalize when the torso deviates from vertical alignment; this involves computation with up_vec
    uprightness_penalty = (1.0 - up_vec[:, 2]).clamp(min=0.0) ** 2
    
    transformed_uprightness_penalty = torch.exp(-uprightness_penalty / torsoness_temp) - 1.0
    
    # Combining into a single consolidated reward
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * transformed_uprightness_penalty
    
    # Collect reward components for diagnostics / analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_uprightness_penalty": transformed_uprightness_penalty
    }
    
    return total_reward, reward_dict
