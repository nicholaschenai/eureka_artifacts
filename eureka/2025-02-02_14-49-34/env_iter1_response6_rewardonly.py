@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Scale and transform distance reward
    temp_distance = 2.0
    distance_reward = 1.0 - torch.exp(-temp_distance * distance_to_drawer)
    
    # Reward for progressing in opening the cabinet door based on the dof position
    temp_progress = 10.0
    progress_reward = torch.tanh(temp_progress * cabinet_dof_pos[:, 3])
    
    # Reward for successful task completion if the drawer is fully opened
    success_threshold = 0.9 # This should be adjusted based on the fully open position
    task_success = (cabinet_dof_pos[:, 3] > success_threshold).float()
    task_reward = task_success * 10.0 # Encourage task completion strongly
    
    # Overall reward
    total_reward = distance_reward + progress_reward + task_reward
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "progress_reward": progress_reward,
        "task_reward": task_reward
    }
    
    return total_reward, reward_components
