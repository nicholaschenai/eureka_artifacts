@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute distance reward as before, reducing weight slightly
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)
    
    # Enhancing open reward with stronger gradient for progress
    temp_open = 2.0  # Increase temperature for a steeper gradient
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    open_reward = open_pos_factor.pow(temp_open)

    # Redefine task success to an intermediate progressive metric
    progressive_completion = torch.where(cabinet_dof_pos[:, 3] > 0.5,
                                         cabinet_dof_pos[:, 3],
                                         torch.zeros_like(cabinet_dof_pos[:, 3]))
    progressive_completion = torch.tanh(progressive_completion)
    
    # Add new task success as more nuanced than pure completion boolean
    task_success_temp = 1.5  # Temperature for final stroke of near-completion
    task_success_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.9, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    task_success_bonus = torch.tanh(task_success_temp * task_success_bonus)

    # Reward re-weighting for contributions
    weight_distance = 0.1
    weight_open = 0.7
    weight_progressive = 0.1
    weight_success = 0.1
    
    total_reward = (weight_distance * distance_reward
                    + weight_open * open_reward
                    + weight_progressive * progressive_completion
                    + weight_success * task_success_bonus)
    total_reward = torch.clamp(total_reward, min=0.0)

    # Reward components tracked
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "progressive_completion": progressive_completion,
        "task_success_bonus": task_success_bonus
    }
    
    return total_reward, reward_components
