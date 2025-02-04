@torch.jit.script
def compute_reward(velocity: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transformation
    velocity_temp = 0.1
    ang_velocity_temp = 0.1

    # Reward for forward velocity (maximize speed)
    forward_velocity_reward = velocity[:, 0]  # Assuming the forward direction is along x-axis
    transformed_forward_velocity_reward = forward_velocity_reward

    # Penalty for high angular velocities (to maintain balance)
    angular_velocity_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    transformed_angular_velocity_penalty = torch.exp(-angular_velocity_penalty / ang_velocity_temp)

    # Combined reward
    total_reward = transformed_forward_velocity_reward + transformed_angular_velocity_penalty
    
    # Return total reward and components
    reward_components = {
        "forward_velocity_reward": transformed_forward_velocity_reward,
        "angular_velocity_penalty": transformed_angular_velocity_penalty
    }

    return total_reward, reward_components
