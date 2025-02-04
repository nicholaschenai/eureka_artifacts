@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    lfinger_pos: torch.Tensor, 
    rfinger_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate the distance between the robot gripper and the drawer
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    # Calculate the middle point between the fingers
    finger_midpoint = (lfinger_pos + rfinger_pos) / 2.0
    # Calculate the alignment of the gripper using distance from the midpoint to grasp position
    gripper_alignment = torch.norm(finger_midpoint - franka_grasp_pos, p=2, dim=-1)
    
    # Reward for minimizing distance to the drawer
    dist_to_drawer_coeff = 1.0
    dist_to_drawer_reward = torch.exp(-dist_to_drawer_coeff * distance_to_drawer)
    
    # Reward for gripper alignment
    alignment_coeff = 2.0
    alignment_reward = torch.exp(-alignment_coeff * gripper_alignment)
    
    # Reward for maximizing the cabinet drawer opening angle
    drawer_open_coeff = 1.0
    drawer_opening_reward = drawer_open_coeff * cabinet_dof_pos[:, 3]
    
    # Total reward composition
    total_reward = dist_to_drawer_reward + alignment_reward + drawer_opening_reward
    
    # Breaking down the reward into components
    reward_components = {
        'dist_to_drawer_reward': dist_to_drawer_reward,
        'alignment_reward': alignment_reward,
        'drawer_opening_reward': drawer_opening_reward
    }
    
    return total_reward, reward_components
