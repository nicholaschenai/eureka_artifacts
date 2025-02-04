@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from Franka's grasp position to drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward based on minimizing distance, now normalized to be more effective
    temp_distance = 0.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Open reward redefined to ensure a clearer gradient, with sigmoid transformation
    door_angle_deg = torch.rad2deg(cabinet_dof_pos[:, 3])  # Assume input in radians
    temp_opening = 0.02
    max_open_deg = 90.0
    normalized_open = door_angle_deg / max_open_deg
    open_reward = torch.sigmoid(temp_opening * (normalized_open - 0.5))  # Transform using sigmoid

    # Combine both rewards, emphasizing the opening reward
    weight_distance = 0.3
    weight_open = 0.7
    total_reward = weight_distance * distance_reward + weight_open * open_reward
    
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
