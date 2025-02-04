@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Adjusted temperature for improved scaling
    temp_distance = 0.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Modify open reward to consider both position increment and velocity
    temp_open = 1.0
    open_reward_pos = cabinet_dof_pos[:, 3]  # position of the drawer
    open_reward_vel = cabinet_dof_vel[:, 3]  # velocity of the drawer opening
    open_reward = torch.sigmoid(temp_open * (open_reward_pos + open_reward_vel))
    
    # Penalty for time spent without progress - keep episodes shorter
    penalize_duration = -0.01
    
    # Total reward is a combination of being close to the drawer, having the door opened, and penalizing delay
    total_reward = distance_reward + open_reward + penalize_duration
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "penalize_duration": penalize_duration
    }
    
    return total_reward, reward_components
