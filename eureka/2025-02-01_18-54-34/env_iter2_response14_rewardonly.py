@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity information
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusted maximum speed for a broader incentive
    max_speed = 20.0
    forward_reward = forward_velocity / max_speed

    # Incorporate a more significant sideways penalty
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * (sideways_velocity / max_speed)

    # Adjust transformations with new temperature parameters
    temperature_forward = 1.5
    temperature_sideways = 0.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * sideways_penalty) - 1.0

    # Heading reward component removed based on feedback

    # Compute total reward considering adjustments and encourage forward running
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and individual component values for tracking
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
