@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    sideways_velocity = velocity[:, 1]

    max_speed = 15.0  # Normalize for appropriate human running speed
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Rewritten sideways penalty
    # Encourage minimal sideways movement by high penalty for high sideways velocity
    tolerant_sideways_velocity = max_speed * 0.1
    rewritten_sideways_penalty = -1.0 * (sideways_velocity.abs() > tolerant_sideways_velocity) * sideways_velocity.abs() / max_speed

    # Temperature parameters
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 0.9
    transformed_rewritten_sideways_penalty = torch.exp(temperature_sideway * rewritten_sideways_penalty) - 1.0

    # Combine reward components
    total_reward = transformed_forward_reward + transformed_rewritten_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "rewritten_sideways_penalty": rewritten_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_rewritten_sideways_penalty": transformed_rewritten_sideways_penalty
    }
