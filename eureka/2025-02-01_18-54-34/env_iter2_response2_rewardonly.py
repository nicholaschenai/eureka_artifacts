@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Increase the scaling to allow faster velocity to count more significantly
    max_speed = 20.0
    forward_reward = torch.clamp(forward_velocity / max_speed, 0, 1)

    # Change sideways penalty to discourage deviation from forward path with more impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = torch.clamp(-5.0 * sideways_velocity / max_speed, -1, 0)

    # Drop the heading reward as it adds no differentiation
    # Remove or adjust transformation for more effective learning
    temperature1 = 1.5
    temperature2 = 0.5
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Construct total reward with more weight on forward velocity
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
