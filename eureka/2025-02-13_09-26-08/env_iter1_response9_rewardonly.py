@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -distance_to_handle
    
    # Stronger reward for actually opening the cabinet door
    door_opening_scale = 10.0
    door_opening_reward = door_opening_scale * cabinet_dof_pos[:, 3]

    # Stronger reward for the velocity of door opening
    door_velocity_scale = 5.0
    door_velocity_reward = door_velocity_scale * cabinet_dof_vel[:, 3]

    # Adjusted transformation for distance reward with a new temperature
    temperature_distance = 0.05  # Adjusted for stronger sensitivity
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # Combine all reward components with adjusted weights
    total_reward = 0.2 * transformed_dist_reward + 1.0 * door_opening_reward + 0.8 * door_velocity_reward

    # Collecting individual components into a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "transformed_distance_reward": transformed_dist_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward
    }

    return total_reward, reward_components
