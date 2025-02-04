@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Increase max speed scaling to emphasize learning fast running
    max_speed = 20.0
    forward_reward = forward_velocity / max_speed

    # Adjust the sideways movement penalty
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Transformed forward reward scaling
    temperature1 = 2.0
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0

    # Adjust transformed sideways penalty for greater emphasis
    temperature2 = 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward with re-scaled components
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return and track individual components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
