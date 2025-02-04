@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Increase and diversify impact of sideways motion for better tuning
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    increased_sideways_penalty = -1.0 * sideways_velocity / max_speed

    # Adjust the transformation temperatures and scales
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * increased_sideways_penalty) - 1.0

    # Combine reward preserving balance between encouragement and penalties
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "increased_sideways_penalty": increased_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
