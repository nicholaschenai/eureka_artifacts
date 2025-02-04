@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward with adjusted reward scaling
    velocity_reward = (potentials - prev_potentials)
    
    # Re-scale the velocity reward and introduce a temperature parameter
    velocity_temp = 0.3  # Reduced temperature to moderate influence
    transformed_velocity_reward = (torch.exp(velocity_reward * velocity_temp) - 1.0) * 0.5  # Re-scale to balance with other rewards

    # Redefine consistency reward to directly measure how aligned the torso is with being upright
    stability_temp = 2.0  # Adjust temperature to increase impact
    upright_penalty = (1.0 - up_vec[:, 2]).abs()
    consistency_reward = torch.exp(-upright_penalty * stability_temp)
    
    # Combine into total reward with balanced weights
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * consistency_reward  # Adjust weights to ensure task completion
    
    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "consistency_reward": consistency_reward
    }

    return total_reward, reward_dict
