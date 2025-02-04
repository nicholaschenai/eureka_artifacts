@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Updated maximum speed for scaling
    max_speed = 20.0
    forward_reward = forward_velocity / max_speed

    # Reevaluate and scale penalty components
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / (max_speed * 2)  # Penalize more heavily

    # Consider dropping heading reward
    # heading_proj = torch.ones_like(forward_velocity)

    # Combining reward - adjusting transformation scales
    temperature1 = 0.5  # Increased positive temperature
    temperature2 = 1.0  # More aggressive penalty temperature
    transformed_forward_reward = torch.tanh(temperature1 * forward_reward)
    transformed_sideways_penalty = torch.tanh(temperature2 * sideways_penalty)

    # Calculating total reward
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return and track individual components without heading reward
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
