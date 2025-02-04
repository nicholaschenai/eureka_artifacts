@torch.jit.script
def compute_reward(hand_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance from the hand to the drawer
    dist_to_drawer = torch.norm(hand_pos - drawer_grasp_pos, dim=-1)
    
    # Reward component for minimizing the distance to the drawer handle
    distance_to_drawer_reward = -dist_to_drawer
    distance_to_drawer_temp = 0.1
    transformed_distance_reward = torch.exp(distance_to_drawer_reward / distance_to_drawer_temp)
    
    # Reward for opening the door (increase in the cabinet's DOF position)
    door_open_reward = cabinet_dof_pos[:, 3]  # assuming 3rd index corresponds to door's DOF position
    door_open_temp = 0.5
    transformed_door_open_reward = torch.exp(door_open_reward / door_open_temp)
    
    # Encourage opening speed
    door_open_speed_reward = cabinet_dof_vel[:, 3]  # similarly assuming 3rd index
    door_open_speed_temp = 0.2
    transformed_door_open_speed_reward = torch.exp(door_open_speed_reward / door_open_speed_temp)

    # Weighted sum of rewards
    total_reward = 0.4 * transformed_distance_reward + 0.4 * transformed_door_open_reward + 0.2 * transformed_door_open_speed_reward
    
    # Create a dictionary for individual rewards
    reward_components = {
        "transformed_distance_reward": transformed_distance_reward,
        "transformed_door_open_reward": transformed_door_open_reward,
        "transformed_door_open_speed_reward": transformed_door_open_speed_reward
    }
    
    return total_reward, reward_components
