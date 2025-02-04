@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0 # Reasonable forward speed limit for normalization
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Rework sideways penalty: Penalize both lateral motion and any downward tilt
    lateral_velocity = velocity[:, 1]
    ang_velocity = root_states[:, 10:13]
    downward_tilt = ang_velocity[:, 2]
    
    sideways_penalty = -0.5 * torch.norm(torch.stack([lateral_velocity, downward_tilt], dim=-1), p=2, dim=-1)
    
    # Temperature adjustments
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * sideways_penalty) - 1.0

    # Total reward calculation
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
