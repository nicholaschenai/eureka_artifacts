@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Emphasize faster running speeds by adjusting scaling and rewards
    max_speed = 20.0  # Assume the agent might achieve even higher speeds
    forward_reward = forward_velocity / max_speed

    # Adjust the sideways penalty scale to balance against larger forward rewards
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / 2.0  # More penalty if deviating sideways

    # Remove constant and unaffected heading reward
    # heading_proj is discarded

    # Adjusted transformations for encouraging running and controlling deviation
    temperature1 = 1.5  # Higher temperature to push forward_velocity rewards
    temperature2 = 0.5  # Penalize sideways motion moderately
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Increase total reward bias towards higher velocity while managing sideways penalty
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return and track individual components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
