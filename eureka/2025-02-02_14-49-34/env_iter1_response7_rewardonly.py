@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward for minimizing the distance to the drawer
    temp_distance = 2.0
    distance_reward = 1.0 - torch.sigmoid(temp_distance * distance_to_drawer)
    
    # Reward for the speed of opening the cabinet door using its velocity
    temp_open_vel = 1.5
    open_speed_reward = torch.sigmoid(temp_open_vel * cabinet_dof_vel[:, 3])
    
    # Penalty for longer episode lengths to encourage shifting towards faster solutions
    episode_length_penalty_weight = -0.01
    episode_length_reward = episode_length_penalty_weight
   
    # Total reward combines different components, balancing proximity and speed of opening
    total_reward = distance_reward + open_speed_reward + episode_length_reward

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_speed_reward": open_speed_reward,
        "episode_length_reward": episode_length_reward
    }
    
    return total_reward, reward_components
