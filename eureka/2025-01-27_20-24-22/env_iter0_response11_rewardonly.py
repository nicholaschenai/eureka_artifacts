@torch.jit.script
def compute_reward(
    hand_pos: torch.Tensor,
    drawer_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    franka_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for transformations
    distance_temp = 0.1
    opening_temp = 5.0
    
    # Reward for minimizing distance to drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, p=2, dim=-1)
    distance_reward = torch.exp(-distance_temp * distance_to_drawer)
    
    # Reward for opening the drawer
    # Here, we assume the drawer opens along the x-axis
    target_drawer_opening_pos = 1.0  # Define a hypothetical target position
    opening_difference = target_drawer_opening_pos - cabinet_dof_pos[:, 0]
    opening_reward = cabinet_dof_pos[:, 0]  # Directly use the first DoF position for drawer opening
    opening_reward = torch.exp(opening_temp * (opening_reward - target_drawer_opening_pos))
    
    # Optional effort penalty (penalize high velocities)
    dof_vel_penalty = 0.0  # Placeholder for any potential penalty computation

    # Total reward
    total_reward = distance_reward + opening_reward - dof_vel_penalty
    
    # Dict for individual reward components
    reward_dict = {
        "distance_reward": distance_reward,
        "opening_reward": opening_reward,
        "dof_vel_penalty": torch.tensor(dof_vel_penalty, device=device)
    }
    
    return total_reward, reward_dict
