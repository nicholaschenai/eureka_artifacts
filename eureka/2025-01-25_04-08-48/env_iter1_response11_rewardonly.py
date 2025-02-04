@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the change in potentials for the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Rescale velocity reward to prevent dominance
    velocity_temp = 0.01
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Improve the stability reward by making it more discriminative
    # Consider rewarding the humanoid for maintaining its upright posture with an exponent
    stability_temp = 1.0
    stability_reward = torch.exp(up_vec[:, 2] * stability_temp) - 1.0

    # Final total reward
    total_reward = transformed_velocity_reward + stability_reward

    # Compose the reward components dictionary
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
