@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute and transform the velocity reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.4  # Updated sensitivity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Implement a new stability reward based on angular velocity
    stability_temp = 0.1  # New temp for stability calculation
    angular_deviation = torch.norm(ang_velocity[:, 0:2], p=2, dim=-1)  # focusing on roll & pitch deviations
    transformed_angular_reward = torch.exp(-angular_deviation * stability_temp)

    # Combine rewards into total reward
    total_reward = 0.7 * transformed_velocity_reward + 0.3 * transformed_angular_reward  # Emphasizing velocity

    # Collect reward components in a dictionary
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_angular_reward": transformed_angular_reward
    }

    return total_reward, reward_dict
