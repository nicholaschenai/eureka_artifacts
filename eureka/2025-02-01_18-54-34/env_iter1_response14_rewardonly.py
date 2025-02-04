@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Adjust max speed for normalization
    max_speed = 12.0  # New value to encourage faster speeds
    forward_reward = forward_velocity / max_speed

    # Penalize sideways and backward movement more significantly
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / (max_speed / 2)  # Strengthen penalty scale

    # Discard constant heading_reward component
    # Optionally we can integrate a new, useful measure if needed in future iterations

    # Transform rewards for stability in training
    temperature1 = 0.6  # New temperature for forward reward transformation
    temperature2 = 0.6  # New temperature for sideways penalty transformation
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
