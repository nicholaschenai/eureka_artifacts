@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Maintain high value for distance reward to emphasize staying close
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # High penalty for being far
    
    # Enhance the door opening reward for clearer task incentives
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    threshold_opening_deg = 45.0
    temp_opening = 0.2
    open_reward = torch.where(door_opening_deg > threshold_opening_deg, torch.ones_like(door_opening_deg), torch.tanh(temp_opening * door_opening_deg))
    
    # Aggregate scaled reward components
    weight_distance = 0.4
    weight_open = 0.6
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Construct reward components dictionary for interpretability later
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
