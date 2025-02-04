@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for minimizing the distance to the drawer handle
    temp_distance = 2.0  # Slightly increased temperature for stronger gradient
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # Applying exponential transformation

    # Reward based on the extent to which the drawer is opened
    temp_open = 0.1  # Adjusted for stronger incentive to open the door
    door_open_extent = cabinet_dof_pos[:, 3]  # Assume this captures the extent the door is open
    max_open_angle = 1.57  # Assume max angle for normalization (90 degrees in radians)
    open_reward = torch.exp(temp_open * (door_open_extent / max_open_angle)) - 1.0  # Encourage fully open condition

    # Encouraging completion of the task
    # If the door is considered 'open', give a significant positive reward
    open_threshold = 1.0
    task_completion_reward = (door_open_extent > open_threshold).float() * 10.0  # Large reward for completion

    # Total reward combines all factors
    total_reward = 0.3 * distance_reward + 0.3 * open_reward + 0.4 * task_completion_reward

    # Define individual reward components for logging
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "task_completion_reward": task_completion_reward
    }
    
    return total_reward, reward_components
