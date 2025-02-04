@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance reward as earlier
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Re-focus and slightly modify the open drawer reward
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)  
    temp_open = 0.5
    open_reward = torch.tanh(temp_open * open_pos_factor)
    
    # New component: Completion bonus when the drawer is fully opened
    temp_completion = 0.3
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.9, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    completion_bonus = torch.tanh(temp_completion * completion_bonus)

    # Sum total reward and ensure all components are non-negative
    weight_distance = 0.2
    weight_open = 0.6
    weight_completion = 0.2
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion * completion_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
