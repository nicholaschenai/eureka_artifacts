@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Enhance forward reward contribution by introducing a higher scaling range
    max_speed = 15.0
    forward_reward = forward_velocity / max_speed

    # Consider refining the sideways penalty by enhancing its sensitivity
    # Rewriting for more dynamic adjustment
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    new_sideways_penalty = torch.clamp(-10.0 * sideways_velocity / max_speed, max=0.0)

    # Removed constant and insignificant heading reward

    # Apply dynamic transformations
    temperature_forward = 0.8  # Slightly reduced to give more gradual incentive
    temperature_sideways = 0.5  # Adjusted to increase penalty impact
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * new_sideways_penalty) - 1.0

    # Adjust total reward combining the revised components
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "new_sideways_penalty": new_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
