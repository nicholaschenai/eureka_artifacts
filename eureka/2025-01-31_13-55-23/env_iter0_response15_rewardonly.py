@torch.jit.script
def compute_reward(
    velocity: torch.Tensor, 
    ang_velocity: torch.Tensor,
    actions: torch.Tensor,
    up_proj: torch.Tensor,
    heading_proj: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Reward based on how fast the ant is moving forward
    velocity_forward_reward = velocity[:, 0]
    
    # Penalty for unnecessary angular velocity (to discourage spinning in place)
    ang_velocity_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    
    # Penalty for excessive actions to promote efficiency
    actions_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Encourage staying upright
    up_projection_reward = up_proj
    
    # Encourage correct heading
    heading_projection_reward = heading_proj
    
    # Normalize and combine components with temperature parameters
    velocity_temp = 1.0
    ang_velocity_temp = 1.0
    actions_temp = 0.005
    up_projection_temp = 1.0
    heading_projection_temp = 1.0
    
    velocity_forward_reward_transformed = torch.exp(velocity_temp * velocity_forward_reward) - 1.0
    ang_velocity_penalty_transformed = torch.exp(-ang_velocity_temp * ang_velocity_penalty)
    actions_penalty_transformed = torch.exp(-actions_temp * actions_penalty)
    up_projection_reward_transformed = torch.exp(up_projection_temp * up_projection_reward) - 1.0
    heading_projection_reward_transformed = torch.exp(heading_projection_temp * heading_projection_reward) - 1.0
    
    total_reward = (
        velocity_forward_reward_transformed
        + ang_velocity_penalty_transformed
        + actions_penalty_transformed
        + up_projection_reward_transformed
        + heading_projection_reward_transformed
    )
    
    reward_components = {
        "velocity_forward_reward": velocity_forward_reward_transformed,
        "ang_velocity_penalty": ang_velocity_penalty_transformed,
        "actions_penalty": actions_penalty_transformed,
        "up_projection_reward": up_projection_reward_transformed,
        "heading_projection_reward": heading_projection_reward_transformed
    }
    
    return total_reward, reward_components
