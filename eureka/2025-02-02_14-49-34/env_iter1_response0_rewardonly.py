@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Reward the agent for minimizing the distance to the drawer
    temp_distance = 0.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Reward for the cabinet door opening, scaled to differentiate progress
    # Now depending on position as well as velocity (smoother trajectory)
    door_pos_reward = cabinet_dof_pos[:, 3]
    temp_open_pos = 2.0
    open_pos_reward = torch.exp(temp_open_pos * door_pos_reward)
    
    door_vel_penalty = torch.abs(cabinet_dof_vel[:, 3])
    temp_open_vel = 1.0
    open_vel_penalty = torch.exp(-temp_open_vel * door_vel_penalty)

    # Total reward combines the new components
    total_reward = distance_reward + open_pos_reward + open_vel_penalty

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_pos_reward": open_pos_reward,
        "open_vel_penalty": open_vel_penalty
    }
    
    return total_reward, reward_components
