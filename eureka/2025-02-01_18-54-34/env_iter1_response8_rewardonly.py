@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Increase the influence of forward reward by scaling
    max_speed = 10.0
    forward_reward = 2.0 * forward_velocity / max_speed  # Multiplied by 2 for increased influence

    # Penalize sideways and backward movement more
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -1.5 * sideways_velocity / max_speed  # Increased penalty scaling

    # Remove or modify constant heading reward as needed
    # Example here: removing the heading reward since it's not changing
    # heading_reward = torch.ones_like(forward_velocity)

    # Transform rewards for stability
    temperature1 = 1.0  # Updated temperature for forward
    temperature2 = 1.0  # Updated temperature for sideways
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
