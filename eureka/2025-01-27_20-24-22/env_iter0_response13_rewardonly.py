@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance to the target (drawer handle)
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=1)
    
    # Calculate the door open extent from the cabinet dof position
    door_open_extent = cabinet_dof_pos[:, 3]

    # Define temperature parameters for transformations
    distance_temp = 0.1
    open_temp = 0.5
    
    # Reward component based on distance to handle
    distance_reward = torch.exp(-distance_temp * distance_to_handle)
    
    # Reward component based on how much the door is open (0 to 1 normalized)
    open_reward = torch.exp(open_temp * (door_open_extent - 1.0))
    
    # Total reward is a combination of both components
    total_reward = distance_reward + open_reward
    
    # Return both total reward and individual components in a dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
