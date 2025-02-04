@torch.jit.script
def compute_reward(velocity: torch.Tensor,
                   ang_velocity: torch.Tensor,
                   up_proj: torch.Tensor,
                   heading_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component : Forward velocity
    forward_vel_reward = velocity[:, 0]
    
    # Reward component : Stability - penalize angular velocity
    ang_vel_penalty = torch.norm(ang_velocity, p=2, dim=-1)

    # Reward component : Uprightness
    up_proj_reward = up_proj

    # Reward component : Heading alignment
    heading_proj_reward = heading_proj

    # Reward calculations
    vel_temp = 1.0
    ang_vel_temp = 5.0
    up_proj_temp = 0.5
    heading_proj_temp = 0.5

    forward_vel_reward = torch.exp(vel_temp * forward_vel_reward)
    ang_vel_penalty = -torch.exp(ang_vel_temp * ang_vel_penalty)
    up_proj_reward = torch.exp(up_proj_temp * (up_proj_reward - 1.0))
    heading_proj_reward = torch.exp(heading_proj_temp * (heading_proj_reward - 1.0))

    total_reward = forward_vel_reward + ang_vel_penalty + up_proj_reward + heading_proj_reward

    reward_dict = {
        'forward_vel_reward': forward_vel_reward,
        'ang_vel_penalty': ang_vel_penalty,
        'up_proj_reward': up_proj_reward,
        'heading_proj_reward': heading_proj_reward
    }

    return total_reward, reward_dict
