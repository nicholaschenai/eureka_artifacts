@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Reward for moving forward; normalize by dividing by a constant max speed value, say 10.0
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Penalize sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Reward or penalize based on stability or heading
    heading_proj = torch.ones_like(forward_velocity)  # Example heading projection placeholder
    heading_reward = heading_proj  # Assuming heading is aligned

    # Combine rewards
    reward = forward_reward + sideways_penalty + 0.1 * heading_reward

    # Transform rewards for stability in training
    temperature1 = 0.5
    temperature2 = 0.5
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "heading_reward": heading_reward,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
