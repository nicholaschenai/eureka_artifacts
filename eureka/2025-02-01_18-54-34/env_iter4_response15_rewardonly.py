@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    
    forward_velocity = velocity[:, 0]
    sideways_velocity = torch.norm(velocity[:, 1:], p=2, dim=-1)

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Redesigned sideways penalty with angular velocity inclusion
    ang_velocity = root_states[:, 10:13]
    sideways_penalty = -0.5 * (sideways_velocity + torch.norm(ang_velocity, p=2, dim=-1)) / max_speed

    # Temperature settings refined for transformations
    temperature_forward = 1.2
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * sideways_penalty) - 1.0

    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,  # Providing the newly defined penalty
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
