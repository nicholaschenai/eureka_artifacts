@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 20.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Increase scale for sideways penalty impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -0.75 * sideways_velocity / max_speed

    # Forward reward transformation with optimized temperature
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Adjust sideways penalty transformation for more impact
    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Recompute total reward to focus more on forward velocity and controlled movement
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
