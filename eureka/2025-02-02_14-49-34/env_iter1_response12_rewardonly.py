@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, prev_cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    # transformed with temperature scaling
    temp_distance = 1.5  # Increased to intensify the focus on minimizing distance
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Reward for opening the cabinet door with a progressive function rather than a constant one
    # transformed with temperature scaling
    temp_open = 2.0
    open_reward = torch.relu(cabinet_dof_pos[:, 3] - prev_cabinet_dof_pos[:, 3]) * torch.exp(temp_open * cabinet_dof_pos[:, 3])

    # Penalty for each time step to encourage faster completion
    time_penalty = -0.005
    
    # Total reward is a combination of rewards and penalty for duration
    total_reward = distance_reward + open_reward + time_penalty
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "time_penalty": torch.tensor(time_penalty, device=franka_grasp_pos.device)
    }
    
    return total_reward, reward_components
