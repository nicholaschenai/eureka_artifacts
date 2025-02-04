@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Give higher reward for reducing distance with new scaling
    temp_distance = 0.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # using exponential decay for better gradient

    # Redefine the open_reward: use polynomial scaling to get a smooth curve of increase
    door_opening = cabinet_dof_pos[:, 3]
    temp_opening = 2.0
    open_reward = torch.pow(torch.clamp(door_opening, 0.0, 1.0), temp_opening)  # Encourage reaching towards full open
    
    # Combine rewards with emphasis on opening
    weight_distance = 0.3
    weight_open = 0.7
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Package the components in a dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
