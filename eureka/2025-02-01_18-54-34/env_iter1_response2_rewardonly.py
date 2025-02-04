@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Enhanced reward for moving forward
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Adjust penalty for sideways movement based on the previously minor impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * sideways_velocity / max_speed  # Increase penalty for sideways movement

    # Revising or discarding heading reward as its contribution is constant and ineffective
    # Use a simpler reward function
    heading_proj = torch.ones_like(forward_velocity)
    # Alternatively, discard heading reward if deemed irrelevant or constant
    # heading_reward = heading_proj

    # Combined raw rewards
    reward = forward_reward + sideways_penalty  # Omit the heading reward for simplicity if it's non-impactful

    # Adjust transformation for rewards
    temperature1 = 0.7  # new temperature for forward reward transformation
    temperature2 = 0.8  # new temperature for sideways penalty transformation
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward with transformations applied
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
