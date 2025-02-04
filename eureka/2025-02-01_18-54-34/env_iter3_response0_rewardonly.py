@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Increase maximum speed to encourage achieving higher velocities
    max_speed = 20.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Update sideways penalty by increasing penalty significance
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -1.0 * sideways_velocity / max_speed

    # Retune transformations based on new analysis
    temperature_forward = 1.0  # Reduced to adjust overly maxing
    transformed_forward_reward = torch.exp(temperature_forward * (forward_reward - 0.5)) - 1.0

    # Adjust sideways penalty transformation
    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Calculate total reward focusing on optimization insights for balance
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
