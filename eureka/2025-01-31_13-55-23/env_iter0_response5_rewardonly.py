@torch.jit.script
def compute_reward(velocity: torch.Tensor, up_proj: torch.Tensor, yaw: torch.Tensor, ang_velocity: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for forward velocity (main goal)
    forward_velocity_reward = velocity[:, 0]  # Assuming positive x-direction is forward
    forward_velocity_temperature = 1.0
    forward_velocity_reward = torch.exp(forward_velocity_temperature * forward_velocity_reward) - 1.0
    
    # Reward component for maintaining upright orientation
    upright_reward = up_proj
    upright_temperature = 1.0
    upright_reward = torch.exp(upright_temperature * upright_reward) - 1.0
    
    # Reward component for minimizing angular velocity (stabilization)
    ang_vel_penalty = torch.sum(torch.abs(ang_velocity), dim=-1)
    ang_vel_temperature = 0.1
    ang_vel_penalty = torch.exp(-ang_vel_temperature * ang_vel_penalty) - 1.0

    # Reward component for minimizing unnecessary actions (energy efficiency)
    action_penalty = torch.sum(torch.abs(actions), dim=-1)
    action_temperature = 0.05
    action_penalty = torch.exp(-action_temperature * action_penalty) - 1.0
    
    # Combine all reward components into the total reward
    total_reward = forward_velocity_reward + upright_reward + ang_vel_penalty + action_penalty
    
    # Create a dictionary for the individual reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "upright_reward": upright_reward,
        "ang_vel_penalty": ang_vel_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_dict
