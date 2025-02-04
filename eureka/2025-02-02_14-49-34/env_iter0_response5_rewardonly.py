@torch.jit.script
def compute_reward(
    drawer_pos: torch.Tensor, 
    franka_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate distance between the gripper and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_pos, dim=-1)
    
    # Encourage the agent to be closer to the drawer handle
    distance_reward_temp = 1.0
    distance_reward = torch.exp(-distance_reward_temp * distance_to_handle)
    
    # Reward for opening the drawer
    # Assuming cabinet_dof_pos corresponds to the openness of the drawer,
    # the higher the value, the more the drawer is open.
    open_reward_temp = 1.0
    open_reward = torch.exp(open_reward_temp * cabinet_dof_pos[:, 0])
    
    # Total reward is a combination of getting close and opening the drawer
    total_reward = distance_reward + open_reward
    
    # Return the total reward and individual components for both the observed reward components
    reward_components = {
        'distance_reward': distance_reward,
        'open_reward': open_reward
    }
    
    return total_reward, reward_components
