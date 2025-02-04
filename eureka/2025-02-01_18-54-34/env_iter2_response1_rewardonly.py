@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Adjust forward velocity scaling to encourage faster speed.
    max_speed = 15.0
    forward_reward = forward_velocity / max_speed

    # Revise sideways penalty for stronger impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -0.5 * sideways_velocity / max_speed

    # Removed heading reward as it wasn't contributing to learning

    # Adjust transformation parameters as necessary
    forward_temperature = 0.5
    sideways_temperature = 0.1
    transformed_forward_reward = torch.exp(forward_temperature * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(sideways_temperature * sideways_penalty) - 1.0

    # Total reward should be focused on speed and penalizing sideways movement
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return individual reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
