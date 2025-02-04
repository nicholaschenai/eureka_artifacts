@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
    previous_cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance to drawer handle
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Transform the distance reward to be sharper
    temp_distance = 1.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Calculate change in cabinet position to reward progression, transformed
    delta_cabinet_pos = cabinet_dof_pos[:, 3] - previous_cabinet_dof_pos[:, 3]
    temp_progress = 2.0
    progress_reward = torch.exp(temp_progress * delta_cabinet_pos)
    
    # Create a new open reward with a better scale and variability
    temp_open = 0.2
    open_reward = torch.exp(temp_open * cabinet_dof_pos[:, 3])
    
    # Introducing penalty for negative velocity (closing movement)
    negative_velocity_penalty = torch.where(cabinet_dof_vel[:, 3] < 0, -0.1, 0.0)

    # Total reward combining all components
    total_reward = distance_reward + open_reward + progress_reward + negative_velocity_penalty
    
    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "progress_reward": progress_reward,
        "negative_velocity_penalty": negative_velocity_penalty
    }
    
    return total_reward, reward_components
