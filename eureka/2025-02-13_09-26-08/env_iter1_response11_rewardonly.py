@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # New reward scale and transformation temperature
    temperature_distance = 0.1  # Temperature for distance transformation
    temperature_door_open = 0.5  # Higher temperature for stronger scaling effect

    # Calculate rewards
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    transformed_dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for door opening (door open position is assumed at index 3)
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / temperature_door_open) - 1.0

    # Penalize door velocity in the wrong direction, reward correct opening
    desired_velocity = 0.02  # A positive small number to ensure the door is opening
    door_velocity_reward = torch.where(cabinet_dof_vel[:, 3] > 0.0, 
                                       cabinet_dof_vel[:, 3] * 0.5, 
                                       -cabinet_dof_vel[:, 3] * 0.5)

    # Combine all reward components
    total_reward = 0.5 * transformed_dist_reward + 2.0 * door_opening_reward + 0.5 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "transformed_distance_reward": transformed_dist_reward,
        "door_opening_reward": door_opening_reward,
        "door_velocity_reward": door_velocity_reward
    }

    return total_reward, reward_components
