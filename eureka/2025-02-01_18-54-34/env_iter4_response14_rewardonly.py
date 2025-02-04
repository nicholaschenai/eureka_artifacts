@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 20.0  # Increased dynamic range for better normalization
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Revised sideways penalty with enhanced scale for greater penalization
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    refined_sideways_penalty = -2.0 * sideways_velocity

    # Temperature control and fine-tuning for improvements
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 0.4
    transformed_sideways_penalty = torch.exp(temperature_sideway * refined_sideways_penalty) - 1.0

    # Compute total reward while ensuring that different components maintain significant relevance
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "refined_sideways_penalty": refined_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
