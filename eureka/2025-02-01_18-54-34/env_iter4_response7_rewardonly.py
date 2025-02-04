@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Re-scale forward reward to have a broader effective range
    max_speed = 15.0
    forward_reward = 2.0 * forward_velocity.clamp(min=0) / max_speed

    # Rewrite sideways penalty with a broader scope for penalty contribution
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    rewritten_sideways_penalty = -1.0 * torch.tanh(sideways_velocity)

    # Transform forward reward with refined temperature
    temperature_forward = 2.0  # Retune for more dynamic forward reward
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Transform sideways penalty with temperature adjustment
    temperature_sideway = 0.8
    transformed_sideways_penalty = torch.exp(temperature_sideway * rewritten_sideways_penalty) - 1.0

    # Compute total reward ensuring balanced contribution from each component
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "rewritten_sideways_penalty": rewritten_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
