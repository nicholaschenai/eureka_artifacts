@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for minimizing the distance to the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = -distance_to_handle
    
    # Transform the distance reward with a new temperature parameter
    temperature_distance = 0.5  # Adjust this parameter as needed
    transformed_distance_reward = torch.exp(distance_reward / temperature_distance)

    # Reward for opening the cabinet door further (scaled to encourage movement)
    door_opening_reward = 5.0 * cabinet_dof_pos[:, 3]  # Rescale door position contribution

    # Reward for positive velocity to encourage opening action
    door_velocity_positive = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    door_velocity_reward = 2.0 * door_velocity_positive

    # Combine reward components
    total_reward = 0.2 * transformed_distance_reward + 1.0 * door_opening_reward + 1.0 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": distance_reward,
        "transformed_distance_reward": transformed_distance_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward
    }

    return total_reward, reward_components
