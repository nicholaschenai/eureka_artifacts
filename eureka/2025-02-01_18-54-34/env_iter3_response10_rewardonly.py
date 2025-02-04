@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed  # Encourage forward velocity

    # Strengthen sideways penalty for better risk avoidance
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -0.75 * sideways_velocity / max_speed

    # Adjust reward transformation strategies
    temperature_forward = 1.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    # Re-design sideways penalty transformation to enhance impact
    temperature_sideway = 2.0
    transformed_sideways_penalty = -torch.abs(torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0)

    # Compose total reward using weighted contributions
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
