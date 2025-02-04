@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    temp_distance = 0.5  # Adjust scale
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # New transformation

    # Reward the agent proportionally to the degree of door opening
    max_open_angle = torch.tensor(1.57)  # 90 degrees in radians, assuming angle range is known
    temp_opening = 0.3  # Adjust scale
    open_reward = cabinet_dof_pos[:, 3] / max_open_angle  # Normalize: a simple linear scale
    open_reward = torch.clamp(open_reward, 0.0, 1.0)
    open_reward = torch.exp(temp_opening * open_reward) - 1  # Exponential reward curve for opening

    # Penalize distance if door is opened less than a threshold
    door_open_threshold = 0.5  # Assume an arbitrary threshold for minimum open angle
    penalized_distance_reward = torch.where(cabinet_dof_pos[:, 3] < door_open_threshold, distance_reward * 0.5, distance_reward)

    # Total reward combines the penalized distance reward and the open reward
    weight_distance = 0.6
    weight_open = 0.4
    total_reward = weight_distance * penalized_distance_reward + weight_open * open_reward

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "penalized_distance_reward": penalized_distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
