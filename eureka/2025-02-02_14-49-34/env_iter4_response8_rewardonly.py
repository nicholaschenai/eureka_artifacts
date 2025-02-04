@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Slightly reduce the scale of the distance reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.8
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)
    
    # Open reward focused on improving the drawer opening state
    temp_open = 0.3
    open_reward = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    open_reward = torch.tanh(temp_open * open_reward)
    
    # Introduce new progress reward: positive reinforcement for any increase in the drawer position
    progress_reward = open_reward * 0.5
    
    # Update the completion bonus condition for achieving near-complete opening
    temp_completion = 0.3
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.9, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    completion_bonus = torch.tanh(temp_completion * completion_bonus)
    
    # Aggregate the total reward with balanced weights
    weight_distance = 0.2
    weight_open = 0.5
    weight_progress = 0.2
    weight_completion = 0.1
    total_reward = (weight_distance * distance_reward + 
                    weight_open * open_reward + 
                    weight_progress * progress_reward + 
                    weight_completion * completion_bonus)
    total_reward = torch.clamp(total_reward, min=0.0)
    
    # Components breakdown for analysis
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "progress_reward": progress_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
