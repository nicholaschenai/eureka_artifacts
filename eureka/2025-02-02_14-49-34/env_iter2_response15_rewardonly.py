@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    temp_distance = 5.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # Exponential scaling for smoother gradient

    # New transformation for open_reward emphasizing velocity: Reward for opening the door
    door_opened_extent = cabinet_dof_pos[:, 3]
    door_opening_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temp_opening = 0.1
    open_reward = torch.sigmoid(door_opened_extent * temp_opening) + torch.sigmoid(door_opening_velocity * temp_opening)  # Combination of position and velocity
    
    # Enhance total reward with increased weights towards door opening success
    weight_distance = 0.3
    weight_open = 0.7
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
