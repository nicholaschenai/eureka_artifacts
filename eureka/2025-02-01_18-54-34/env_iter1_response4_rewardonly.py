@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # New Reward for moving forward
    forward_reward = forward_velocity / 8.0  # Slightly lower max speed assumption for higher reward values

    # Discarding sideways penalty component as it's nearly constant
    # New Reward for maintaining heading (adjusting based on alignment)
    heading_proj = torch.ones_like(forward_velocity)  # Reassessing heading projection placeholder
    heading_reward_weight = 0.2
    heading_reward = heading_proj * heading_reward_weight  # Assuming hitting target projection is beneficial

    # Combine rewards with adjusted scaling
    reward = forward_reward + heading_reward

    # Transform rewards for stability in training with tweaked temperatures
    temperature1 = 0.7
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0

    # Total reward calculation
    total_reward = transformed_forward_reward

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "heading_reward": heading_reward,
        "transformed_forward_reward": transformed_forward_reward,
    }
