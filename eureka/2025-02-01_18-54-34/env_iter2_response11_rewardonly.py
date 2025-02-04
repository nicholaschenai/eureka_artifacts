@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Revise maximum speed for scaling
    max_speed = 15.0
    forward_reward = forward_velocity / max_speed

    # Rebalanced sideways penalty component
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Discarded constant heading reward

    # Compute transformed components with adjusted temperature
    temperature1 = 1.2  # Adjusted temperature for forward reward
    temperature2 = 0.5  # Adjusted temperature for sideways penalty reduction
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Update total reward with continued emphasis on forward momentum
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the rewritten reward function incorporating only effective components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
