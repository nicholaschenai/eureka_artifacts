@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Increased weight on ~sideways_penalty by revising scale
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    scaled_sideways_penalty = -1.0 * sideways_velocity / max_speed

    # Transformations tuned for impact
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * scaled_sideways_penalty) - 1.0

    # Combined important features
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "scaled_sideways_penalty": scaled_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
