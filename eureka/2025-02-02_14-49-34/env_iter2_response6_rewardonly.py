@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters
    temp_distance = 2.0
    temp_opening = 0.1
    temp_vel_penalty = 0.5

    # Calculating the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward for minimizing the distance to the drawer's grasp position
    # Apply exponential scaling for refinement
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)

    # Reward for opening the cabinet
    # Assume max_open_deg indicates the full opening angle; replace with actual if available
    max_open_deg = 90.0
    open_reward = torch.sin(temp_opening * torch.rad2deg(cabinet_dof_pos[:, 3]) / max_open_deg)

    # Penalty for not moving/opening the cabinet
    vel_penalty = torch.exp(-temp_vel_penalty * torch.abs(cabinet_dof_vel[:, 3]))

    # Total reward is a balance between reaching, opening, and penalizing inactivity
    weight_distance = 0.4
    weight_open = 0.4
    weight_vel = 0.2
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_vel * vel_penalty
    
    # Construct reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "vel_penalty": vel_penalty
    }
    
    return total_reward, reward_components
