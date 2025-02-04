@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent more substantially for reducing the distance, applying exponential transformation
    temp_distance = 2.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Reward for opening the drawer, but giving higher reward for larger openings
    door_angle_deg = torch.rad2deg(cabinet_dof_pos[:, 3])  # Assume radian inputs
    max_open_deg = 90.0  # Maximum assumed open angle
    # Use a temperature parameter to scale the sine transformation
    temp_opening = 10.0 / max_open_deg
    open_reward = torch.sin(temp_opening * door_angle_deg) ** 2

    # Combine the rewards, emphasizing the task completion more to improve task score
    weight_distance = 0.3
    weight_open = 0.7
    total_reward = weight_distance * distance_reward + weight_open * open_reward
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
