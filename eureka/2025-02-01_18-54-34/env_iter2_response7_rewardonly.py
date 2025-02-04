@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary state information
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Adjust scaling for maximum speed to highlight fast running
    max_speed = 18.0
    forward_reward = forward_velocity / max_speed

    # Rework sideways penalty for greater impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Temporarily remove heading reward 
    # Act on heading_penalty previously for balance/noise
    # heading_proj = torch.ones_like(forward_velocity)

    # Revised reward transformation with adjusted temperatures
    temperature1 = 1.5
    temperature2 = 0.4
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Ensure the total reward strongly favors forward movement
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return total reward tracked with individual components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        # Removed heading_reward in this iteration
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
