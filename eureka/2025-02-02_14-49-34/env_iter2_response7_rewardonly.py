@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reduce emphasis on the distance reward by transforming with new scale
    temp_distance = 2.0
    distance_reward = 1.0 / (1.0 + temp_distance * distance_to_drawer)  # Exponential decay for distance influence

    # Prioritize the opening of the door with a refined transformation
    door_angle_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    max_open_deg = 90.0  # Assume achievable max open angle
    temp_opening = 0.15
    open_reward = torch.exp(-torch.square(temp_opening * (max_open_deg - door_angle_deg)))  # Bell curve for successful opening

    # Total reward prioritizes open_reward and balances distance_reward component
    weight_distance = 0.3
    weight_open = 0.7
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Clamp open_reward to prevent negative contributions
    open_reward = torch.clamp(open_reward, min=0.0)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
