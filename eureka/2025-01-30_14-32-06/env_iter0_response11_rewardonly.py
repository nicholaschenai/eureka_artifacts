@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack velocity from root_states
    velocity = root_states[:, 7:10]

    # Calculate forward velocity (assuming it's along the x-axis)
    forward_velocity = velocity[:, 0]

    # Incentivize forward velocity
    velocity_reward = forward_velocity

    # Encourage proximity to the target
    target_proximity_reward = potentials - prev_potentials

    # Normalize rewards
    temp_velocity = 1.0  # Temperature for velocity reward normalization
    temp_proximity = 1.0  # Temperature for target proximity reward normalization

    velocity_reward_transformed = torch.exp(forward_velocity / temp_velocity) - 1.0
    target_proximity_reward_transformed = torch.exp(target_proximity_reward / temp_proximity) - 1.0

    # Total reward
    total_reward = velocity_reward_transformed + target_proximity_reward_transformed

    reward_components = {
        "velocity_reward": velocity_reward_transformed,
        "target_proximity_reward": target_proximity_reward_transformed
    }

    return total_reward, reward_components
