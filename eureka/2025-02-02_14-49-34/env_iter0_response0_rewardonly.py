@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   drawer_rot: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate distance between the gripper and the drawer handle
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Reward for minimizing the distance to the drawer handle
    distance_reward_temperature = 10.0
    distance_reward = torch.exp(-distance_reward_temperature * distance_to_drawer)
    
    # Reward for rotating the drawer (opening it)
    # Assuming the rotation in the y-axis indicates the door opening
    door_open_angle = cabinet_dof_pos[:, 3]  # Index 3 is specific to the given obs_buf setup
    rotation_reward_temperature = 1.0
    rotation_reward = torch.exp(rotation_reward_temperature * door_open_angle)
    
    # Sum the rewards with weighted factors
    total_reward = 0.6 * distance_reward + 0.4 * rotation_reward
    
    return total_reward, {'distance_reward': distance_reward, 'rotation_reward': rotation_reward}
