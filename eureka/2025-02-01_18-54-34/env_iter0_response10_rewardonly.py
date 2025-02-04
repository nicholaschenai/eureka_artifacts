@torch.jit.script
def compute_reward(
    velocity: torch.Tensor, 
    up_proj: torch.Tensor, 
    ang_velocity: torch.Tensor, 
    dt: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward components
    vel_reward_temp = 1.0
    up_proj_reward_temp = 0.5
    ang_velocity_penalty_temp = 0.2
    
    # Reward for forward velocity
    velocity_reward = velocity[:, 0]  # Assuming forward direction is the x-axis
    velocity_reward_transformed = torch.exp(velocity_reward / vel_reward_temp)

    # Reward for maintaining upright position
    up_proj_reward = up_proj
    up_proj_reward_transformed = torch.exp(up_proj_reward / up_proj_reward_temp)

    # Penalty for high angular velocities (for stability)
    ang_velocity_magnitude = torch.norm(ang_velocity, p=2, dim=-1)
    ang_velocity_penalty = torch.exp(-ang_velocity_magnitude / ang_velocity_penalty_temp)

    # Total reward
    total_reward = velocity_reward_transformed + up_proj_reward_transformed + ang_velocity_penalty

    # Reward components dictionary
    reward_components = {
        "velocity_reward": velocity_reward_transformed,
        "up_proj_reward": up_proj_reward_transformed,
        "ang_velocity_penalty": ang_velocity_penalty
    }
    
    return total_reward, reward_components
