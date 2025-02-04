@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Re-adjust forward reward; increase its weight
    max_speed = 15.0  # Increase max speed to explore more velocity range
    forward_reward = forward_velocity / max_speed

    # Increase penalty for sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * sideways_velocity / max_speed  # Increase the penalty weight

    # Consider removing the heading_reward as it provides no varying learning signal
    # heading_proj = torch.ones_like(forward_velocity) 
    # heading_reward = heading_proj  

    # Combine rewards; give more weight to the forward_reward
    reward = forward_reward + sideways_penalty

    # Transform rewards with different temperatures
    temperature1 = 0.3  # Adjust temperature for forward
    temperature2 = 0.7  # Adjust temperature for sideways
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward calculation
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
