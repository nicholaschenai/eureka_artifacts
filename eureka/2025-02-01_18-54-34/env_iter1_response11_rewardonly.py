@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Improved reward for moving forward; more significant scaling
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed
    transformed_forward_reward = (forward_reward * 2.0).clamp(0, 1.0) # Scale to 0-1 range
    
    # Adjust sideways penalty scaling for optimization
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed
    transformed_sideways_penalty = sideways_penalty.clamp(-1.0, 0) * 0.5  # Adjust scale
    
    # Refactor or discard heading_reward as it was ineffective
    heading_proj = torch.ones_like(forward_velocity)  # Placeholder
    heading_reward = heading_proj * 0.0  # Effectively neutral

    # Combine rewards
    reward = transformed_forward_reward + transformed_sideways_penalty + heading_reward

    # Total reward and individual components
    return reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty,
        "heading_reward": heading_reward,
    }
