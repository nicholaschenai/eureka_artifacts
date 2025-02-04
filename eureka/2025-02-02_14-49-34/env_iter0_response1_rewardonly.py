@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    # transformed with temperature scaling
    temp_distance = 1.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Reward for the cabinet door opening, assuming higher dof_pos means more open
    # transformed with temperature scaling
    temp_open = 0.5
    open_reward = torch.exp(temp_open * cabinet_dof_pos[:, 3])  # considering only one dof related to the door

    # Total reward is a combination of being close to the drawer and having the door opened
    total_reward = distance_reward + open_reward
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
