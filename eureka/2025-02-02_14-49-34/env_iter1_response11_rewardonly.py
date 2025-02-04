@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # New scaling for distance reward
    temp_distance = 0.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Open reward - reward for increasing dof position incrementally
    drawer_opening = cabinet_dof_pos[:, 3]  # Using the dof pos related to the door
    temp_open = 1.0  # Larger scaling for incremental encouragement
    open_reward = torch.exp(temp_open * (drawer_opening - 0.5)) - 1  # Shifting to normalize

    # Success reward
    success_threshold = 0.8
    success_reward = torch.where(drawer_opening > success_threshold, torch.tensor(1.0, device=drawer_opening.device), torch.tensor(0.0, device=drawer_opening.device))
    
    # Combine all rewards
    total_reward = distance_reward + open_reward + 0.1 * success_reward  # Weighted success reward
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "success_reward": success_reward
    }
    
    return total_reward, reward_components
