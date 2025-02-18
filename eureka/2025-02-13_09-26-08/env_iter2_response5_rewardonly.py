@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance reward with adjusted scaling
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)
    
    # Enhanced reward for progressively opening the drawer
    door_open_amount = cabinet_dof_pos[:, 3].clamp(min=0.0)  # Ensure no negative values
    door_open_reward = door_open_amount * 3.0  # Increased scaling to prioritize this task
    
    # Feedback for velocity aiding in opening the drawer
    door_velocity_positive = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    velocity_reward = door_velocity_positive * 2.0  # Rescaled for greater impact

    # Penalizing unnecessary movements by reducing reward if the policy reaches very high episode length
    penalty_for_long_episodes = 0.01  # Small penalty for every prolonged episode step
    
    # Sum of all rewards and penalty
    total_reward = dist_reward + door_open_reward + velocity_reward - penalty_for_long_episodes
    
    # Compose outputs in dictionaries for tracking purposes
    reward_components = {
        "dist_reward": dist_reward,
        "door_open_reward": door_open_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
