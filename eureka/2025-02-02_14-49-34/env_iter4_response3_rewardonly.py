@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to drawer position for grasping
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 1.0  # Adjust scaling for clarity
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Encourage opening drawer by estimating the 'open state'
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 1.0  # More aggressive encouragement
    open_reward = torch.exp(open_pos_factor * temp_open) - 1.0  # Transform for non-linearity to push opening
    
    # Better success criteria if fully opened
    completion_success_threshold = 0.95
    completion_criterion = (cabinet_dof_pos[:, 3] >= completion_success_threshold).float()
    completion_bonus = completion_criterion  # Clear cutoff rewards full opening

    # Combine weighted values while controlling the influence scopes
    weight_distance = 0.1
    weight_open = 0.7
    weight_completion = 0.2
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion * completion_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Reward components breakdown for further analysis
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus
    }

    return total_reward, reward_components
