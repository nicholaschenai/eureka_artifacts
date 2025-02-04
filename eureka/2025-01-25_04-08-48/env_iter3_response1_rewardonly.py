@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Adjust velocity reward with a temperature parameter to moderate its influence
    velocity_temp = 0.1  # Lowered to reduce dominance
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefine consistency reward to incorporate angular stability
    ang_stability_temp = 2.0  # Higher temperature for sensitivity
    ang_stability = torch.exp(-torch.norm(root_states[:, 10:13], p=2, dim=-1) * ang_stability_temp)

    # Combine rewards into a total reward
    total_reward = 0.7 * transformed_velocity_reward + 0.3 * ang_stability  # Rebalanced reward weights

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "angular_stability_reward": ang_stability
    }

    return total_reward, reward_dict
