@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Adjust velocity reward with a temperature parameter
    velocity_temp = 0.5  # Increased temperature for higher sensitivity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Introduce a redefined stability reward using uprightness and velocity consistency
    stability_temp = 1.0  # Adjusted temperature for balancing influence
    uprightness = (up_vec[:, 2] - 1.0).abs()
    consistency_reward = torch.exp(-uprightness * stability_temp) 

    # Combine into total reward
    total_reward = 0.8 * transformed_velocity_reward + 0.2 * consistency_reward  # Balancing components

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "consistency_reward": consistency_reward
    }

    return total_reward, reward_dict
