@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Rescale maximum desirable speed to encourage higher speeds
    max_speed = 20.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed * 2

    # Sideways penalty with a sharper scaling for better discouragement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -sideways_velocity / max_speed

    # Stronger transformation for forward reward with adjusted temperature
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    # Stronger transformation for sideways penalty with adjusted temperature
    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Recalculate total reward with enhanced penalty
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
