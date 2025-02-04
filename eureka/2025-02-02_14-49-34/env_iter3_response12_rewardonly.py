@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Normalized and inverted distance reward to encourage hand to be near the drawer
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)
   
    # Reward for opening the cabinet door using a sensitive function
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.15
    open_reward = torch.tanh(temp_opening * door_opening_deg)
    
    # Reworking speed reward to only reward successful opening
    velocity_reward = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    temp_velocity = 0.5
    speed_reward = torch.tanh(temp_velocity * velocity_reward)

    # Additional component: Closer proximity or contact reward to encourage reaching
    temp_contact = 2.0
    contact_reward = torch.exp(-temp_contact * distance_to_drawer)
    
    # Combine the rewards into a total, weighted sum
    weight_distance = 0.2
    weight_open = 0.4
    weight_speed = 0.2
    weight_contact = 0.2
    total_reward = (weight_distance * distance_reward 
                    + weight_open * open_reward 
                    + weight_speed * speed_reward 
                    + weight_contact * contact_reward)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "contact_reward": contact_reward
    }
    
    return total_reward, reward_components
