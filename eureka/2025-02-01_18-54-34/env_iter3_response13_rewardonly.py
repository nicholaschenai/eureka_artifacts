@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed * 2.0  # Increased scaling factor for more influence

    # Rescale sideways penalty for greater impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -0.8 * sideways_velocity / max_speed  # Adjusted scaling for stronger penalty

    # Adjust transformed components further
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Increase scaling for transformed sideways penalty
    temperature_sideway = 0.8
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Recalculate total reward with realigned emphasis
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
