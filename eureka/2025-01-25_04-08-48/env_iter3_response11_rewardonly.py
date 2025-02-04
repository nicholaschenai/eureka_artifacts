@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Adjust velocity reward with a temperature parameter
    velocity_temp = 0.4  # Adjusted downwards for balancing impact
    transformed_velocity_reward = 1 - torch.exp(-velocity_reward * velocity_temp)

    # Redefine consistency reward to focus directly on maintaining a target upright orientation
    uprightness_temp = 2.0  # Increased temperature for sharper sensitivity
    target_uprightness = 1.0  # Perfectly upright as target
    uprightness_error = (up_vec[:, 2] - target_uprightness).abs()
    consistency_reward = 1 - torch.exp(-uprightness_error * uprightness_temp)

    # Combine and balance total reward components
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * consistency_reward  # Equal weighting to incentivize balance and speed

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "consistency_reward": consistency_reward
    }

    return total_reward, reward_dict
