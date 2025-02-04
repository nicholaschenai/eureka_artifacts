@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Initial scaling for the forward reward
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Penalize sideways velocity more aggressively due to its stability in training
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / (max_speed / 2)  # Double the penalty effect

    # Remove the redundant constant heading reward component
    # heading_proj = torch.ones_like(forward_velocity)  # No longer needed

    # Updated transformation parameters
    temperature1 = 0.5
    temperature2 = 0.4
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Adjust the weightings to focus on forward movement
    total_reward = transformed_forward_reward + 0.5 * transformed_sideways_penalty

    # Return and track individual components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
