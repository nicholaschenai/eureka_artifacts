@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Recalculate distance to handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -distance_to_handle

    # Enhanced reward component for actually opening the cabinet door
    door_opening_reward = torch.clip(cabinet_dof_pos[:, 3], 0.0, 1.0) * 2.0  # Improved scale

    # Enhanced reward component for positive velocity towards opening
    door_velocity_reward = torch.clip(cabinet_dof_vel[:, 3], 0.0, None) * 1.5  # Improved scale

    # Adjust the transformation for distance reward using a revised temperature parameter
    temperature_distance = 0.2  # Adjusted temperature for lesser dominance
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # Rebalance the total reward with newly scaled components
    total_reward = 0.3 * transformed_dist_reward + 1.0 * door_opening_reward + 0.7 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "transformed_distance_reward": transformed_dist_reward
    }

    return total_reward, reward_components
