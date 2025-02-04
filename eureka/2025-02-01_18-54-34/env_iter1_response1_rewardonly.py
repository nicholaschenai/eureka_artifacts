@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Adjusted reward scaling for forward movement
    max_speed = 12.0  # slightly increased
    forward_reward = forward_velocity / max_speed

    # Penalize sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Consider dropping original heading reward due to lack of variance
    # We can discard or redefine the component if it provides no useful variance

    # Combine rewards
    reward = forward_reward + sideways_penalty

    # Apply transformation for normalized reward
    temperature1 = 0.5
    temperature2 = 0.4
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
