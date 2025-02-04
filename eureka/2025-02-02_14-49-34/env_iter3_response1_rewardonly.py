@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Reward for opening the cabinet door
    door_angle = cabinet_dof_pos[:, 3] # Assuming angle in radians
    target_open_angle = 1.0 # Assuming 1 radian as the fully open angle
    temp_open = 1.0
    open_reward = torch.exp(-temp_open * torch.abs(door_angle - target_open_angle))
    
    # Task success reward: when the door is fully open
    success_reward = torch.where(door_angle >= target_open_angle, torch.tensor(1.0, device=franka_grasp_pos.device), torch.tensor(0.0, device=franka_grasp_pos.device))
    
    # Combine rewards
    weight_distance = 0.2
    weight_open = 0.6
    weight_success = 0.2
    total_reward = (weight_distance * distance_reward + 
                    weight_open * open_reward + 
                    weight_success * success_reward)
    
    # Create reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "success_reward": success_reward
    }
    
    return total_reward, reward_components
