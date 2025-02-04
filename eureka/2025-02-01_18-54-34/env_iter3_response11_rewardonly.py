@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed
    
    # Increased penalty for sideways velocity
    sideways_velocity = torch.abs(velocity[:, 1])
    optimized_sideways_penalty = -1.0 * sideways_velocity / max_speed

    # Adjust forward transformation temperature to enhance differentiation
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    # Increase sideway penalty transformation to push more constraint
    temperature_sideway = 2.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    total_reward = transformed_forward_reward + 0.5 * transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
