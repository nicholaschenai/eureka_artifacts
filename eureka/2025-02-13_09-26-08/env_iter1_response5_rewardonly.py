@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Component 1: Distance to Handle (Encourage reaching)
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -distance_to_handle

    # Component 2: Door Opening (Encourage opened position)
    door_target_open_angle = torch.tensor(1.0, device=cabinet_dof_pos.device)  # hypothetical target position
    door_open_progress = torch.abs(cabinet_dof_pos[:, 3] - door_target_open_angle)
    door_opening_reward = torch.exp(-door_open_progress)  # Transform this for smooth gradient

    # Component 3: Door Velocity (Encourage positive velocity for opening motion)
    door_velocity_threshold = 0.1  # Encourage significant velocity
    door_velocity_reward = (cabinet_dof_vel[:, 3] > door_velocity_threshold).float()

    # Transform the distance reward using a temperature parameter
    temperature_distance = 0.2  # Adjusted for sensitivity
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)
    
    # Sum and weighted combination of rewards
    total_reward = 0.3 * transformed_dist_reward + 0.4 * door_opening_reward + 0.3 * door_velocity_reward
    
    # Track individual reward components
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "transformed_distance_reward": transformed_dist_reward
    }

    return total_reward, reward_components
