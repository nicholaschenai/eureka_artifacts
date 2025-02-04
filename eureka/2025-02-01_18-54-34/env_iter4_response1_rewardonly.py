@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 20.0  # Adjusting to a more extensive forward speed for normalization
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Redefine sideways penalty with angular velocity penalty to reduce deviation from forward motion
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    ang_velocity = root_states[:, 10:13]
    angular_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    normalized_sideways_penalty = -(0.4 * sideways_velocity + 0.3 * angular_penalty) / max_speed

    # Apply exponential transformation with temperature control
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Strengthen sideways penalty through new definition and transformation
    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(normalized_sideways_penalty * temperature_sideway) - 1.0

    # Compute total reward, balancing optimized components
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "normalized_sideways_penalty": normalized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
