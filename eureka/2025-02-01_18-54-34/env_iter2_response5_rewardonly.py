@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Re-calibrating the maximum forward speed assumption
    max_speed = 20.0  # Increased assumed max speed for better scaling
    forward_reward = forward_velocity / max_speed

    # Re-evaluating sideways penalty for greater emphasis
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / (0.5 * max_speed)

    # Dropping the constant heading reward
    # heading_proj should be considered null

    # Update transformed rewards scaling
    temperature_forward = 1.5
    temperature_sideways = 0.5  # Adjusting temperatures for effects
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = -torch.exp(temperature_sideways * sideways_penalty) + 1.0

    # Total reward optimization balance
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Providing detailed component logging
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
