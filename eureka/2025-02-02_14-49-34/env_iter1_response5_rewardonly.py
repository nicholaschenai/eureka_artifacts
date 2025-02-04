@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    # transformed with temperature scaling
    temp_distance = 1.0
    distance_reward = 1.0 - torch.clamp(distance_to_drawer, 0.0, 1.0)  # Normalize and invert to give more gradient
    
    # Redefine the open_reward: increase with how much the door is opened
    # Use sine scaling to enhance sensitivity
    door_angle_deg = torch.rad2deg(cabinet_dof_pos[:, 3])  # Assume radian inputs
    max_open_deg = 90.0  # Specify an assumption of maximum open angle
    temp_opening = 0.05
    open_reward = torch.sin(temp_opening * door_angle_deg)  # Try a sinusoidal transformation to open_doors
    
    # Total reward is a normalized combination of the two components
    weight_distance = 0.5
    weight_open = 0.5
    total_reward = weight_distance * distance_reward + weight_open * open_reward
    
    # Clamping open_reward to be non-negative
    open_reward = torch.clamp(open_reward, min=0.0)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
