@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved distance reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3
    distance_reward = 1.0 - torch.exp(-temp_distance * distance_to_drawer)

    # Emphasized and rescaled open reward
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open_reward = 1.0
    open_reward = torch.exp(temp_open_reward * open_pos_factor) - 1.0

    # Incorporate task-specific score (e.g., a movement threshold for the opening)
    task_completion_condition = torch.gt(cabinet_dof_pos[:, 3], 0.8)
    task_completion_bonus = torch.where(task_completion_condition, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    temp_task_bonus = 0.5
    task_score = torch.exp(temp_task_bonus * task_completion_bonus) - 1.0

    # Total reward calculation with adjusted weights
    weight_distance = 0.4
    weight_open = 0.5
    weight_completion = 0.1
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion * task_score

    # Clamp the total reward to be non-negative
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "task_score": task_score
    }

    return total_reward, reward_components
