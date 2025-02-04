@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Improved reward for moving forward; adjust by dividing by a higher max speed value
    max_speed = 15.0
    forward_reward = forward_velocity / max_speed

    # Higher penalty for sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Introduce a stability/heading alignment factor and discard heading_reward as it was ineffective
    tors_rotation_stability = torch.ones_like(forward_velocity)  # Placeholder for stability measure
    stability_reward = torch.clamp(tors_rotation_stability, 0.0, 1.0)  # Ensure within [0, 1]

    # Apply transformation functions with tuned temperature values
    temperature1 = 0.7
    temperature2 = 0.3
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward combining the adjustments
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "stability_reward": stability_reward,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
