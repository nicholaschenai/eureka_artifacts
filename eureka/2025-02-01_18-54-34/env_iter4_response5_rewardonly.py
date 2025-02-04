@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Normalize forward velocity to encourage fast running
    max_speed = 15.0  # Expected maximum running speed
    forward_reward = forward_velocity.clip(min=0) / max_speed
    
    # Rewritten sideways penalty - penalize deviation from desired forward axis
    sideways_velocity = torch.norm(velocity[:, 1:], p=2, dim=-1)
    optimized_sideways_penalty = -torch.pow(sideways_velocity, 1.5) / max_speed

    # Temperature-based transformation for forward reward to enhance differentiation
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Remove transform from sideways penalty: Penalize directly for greater impact
    total_reward = transformed_forward_reward + optimized_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
    }
