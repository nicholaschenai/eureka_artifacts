@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Velocity Reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.5  # Increased temperature for stronger effect
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Revised Stability Reward
    stability_measure = (up_vec[:, 2] - 1.0).abs()
    stability_temp = 5.0  # Adjusted temperature for sensitiveness
    transformed_stability_reward = torch.exp(-stability_measure * stability_temp)  # Encourage stable posture

    # Optional New Component: Forward Velocity Consistency
    forward_vel_consistency = velocity[:, 0]  # Considering x-axis as forward direction
    forward_temp = 1.0
    forward_consistency_reward = torch.exp(forward_vel_consistency * forward_temp) - 1.0

    # Total Reward: Rebalance emphasis between components
    total_reward = 1.2 * transformed_velocity_reward + 0.3 * transformed_stability_reward + 0.5 * forward_consistency_reward

    # Collect individual rewards for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
        "forward_consistency_reward": forward_consistency_reward
    }

    return total_reward, reward_dict
