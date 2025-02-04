@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Reward for moving forward; normalize by dividing by a constant max speed value, say 10.0
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Increase penalty for sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * sideways_velocity / max_speed

    # Remove or reformulate constant heading_reward
    # Assuming constant heading_reward might not be useful, it is discarded
    
    # Combine rewards
    reward = forward_reward + sideways_penalty

    # Adjust transformations to provide more gradient
    temperature_forward = 1.0
    temperature_sideways = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * sideways_penalty) - 1.0

    # Total reward with more emphasis on forward_reward after transformation
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
