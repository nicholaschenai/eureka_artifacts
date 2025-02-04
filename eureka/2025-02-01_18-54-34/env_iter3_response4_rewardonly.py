@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusting scale for max_speed to enhance forward velocity reward potential
    max_speed = 20.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Rewriting sideways penalty for larger impact on policy optimization
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -0.1 * sideways_velocity / max_speed

    # Optimize reward transformations for broader planning scale
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    temperature_sideway = 0.3
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Strengthen reward components against side penalties
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
