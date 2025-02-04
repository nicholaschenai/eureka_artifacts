@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance reward with a lowered weight
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.2
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Rescale and enhance open reward
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 2.0  # Increased for stronger emphasis
    open_reward = torch.exp(temp_open * open_pos_factor) - 1
    
    # Define clearer completion bonus for task success
    completion_threshold = 0.5  # Lowered for testing partial success
    completion_bonus = torch.where(open_pos_factor > completion_threshold, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    
    # Define a task score for opening and holding the drawer in a successful position
    temp_task = 1.5  # For stronger long-term success emphasis
    task_score = torch.exp(temp_task * (open_pos_factor - completion_threshold)) - 1

    # Sum total reward and prioritize certain components
    weight_distance = 0.1
    weight_open = 0.5
    weight_completion = 0.3
    weight_task = 0.1
    total_reward = (
        weight_distance * distance_reward +
        weight_open * open_reward +
        weight_completion * completion_bonus +
        weight_task * task_score
    )
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus,
        "task_score": task_score
    }
    
    return total_reward, reward_components
