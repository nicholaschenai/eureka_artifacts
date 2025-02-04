@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    max_speed = 15.0
    forward_reward = forward_velocity / max_speed

    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * (sideways_velocity / max_speed)  # Slightly increase penalty importance

    # Eliminating constant heading reward because it provides no dynamic value
    normalized_forward_reward = forward_velocity / torch.abs(velocity).sum(dim=-1, keepdim=True).clamp(min=1.0)
    
    # Adjust transformations with new temperatures
    temperature_forward = 1.5
    temperature_sideways = 0.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * sideways_penalty) - 1.0

    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "normalized_forward_reward": normalized_forward_reward,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
