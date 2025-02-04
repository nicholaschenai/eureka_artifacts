@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, contact_force_scale: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity (assuming forward is along the x-axis)
    velocity = root_states[:, 7]  # forward velocity (x-axis component)

    # Reward for moving forward
    forward_velocity_reward = velocity
    # Clip negative values since we only want positive forward movement
    forward_velocity_reward = torch.clamp(forward_velocity_reward, min=0.0)

    # Reward for staying upright (using the up vector's component along the positive z-axis)
    upness_reward = up_vec[:, 2]  # Assume up_vec's z-component defines being upright

    # Penalty for high force usage (actions represent forces/torques, scaled)
    force_penalty = torch.sum(actions ** 2, dim=-1) * contact_force_scale
    force_penalty = torch.clamp(force_penalty, min=0.0)

    # Calculate total reward with weights
    total_reward = forward_velocity_reward * 1.0 + upness_reward * 0.1 - force_penalty * 0.001

    # Temperature parameter for reward transformations
    velocity_temp = 1.0
    upness_temp = 0.1
    force_temp = 0.001

    # Transform rewards
    transformed_forward_velocity_reward = torch.exp(forward_velocity_reward / velocity_temp)
    transformed_upness_reward = torch.exp(upness_reward / upness_temp)
    transformed_force_penalty = torch.exp(-force_penalty / force_temp)

    # Compose final reward
    total_reward = transformed_forward_velocity_reward + transformed_upness_reward - transformed_force_penalty

    # Individual rewards dictionary
    reward_components = {
        "forward_velocity_reward": transformed_forward_velocity_reward,
        "upness_reward": transformed_upness_reward,
        "force_penalty": transformed_force_penalty
    }

    return total_reward, reward_components
