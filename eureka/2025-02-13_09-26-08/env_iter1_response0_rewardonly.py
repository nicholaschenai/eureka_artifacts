@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute distance to handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Reward for minimizing the distance to the handle, using a stronger transformation
    temperature_distance = 0.05  # Tuned temperature for transformation
    transformed_dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for the door opening based on hinge joint position
    door_opening_reward = torch.clip(cabinet_dof_pos[:, 3], min=0.0)  # Clip to avoid negative rewards

    # Reward to encourage the velocity in the opening direction
    door_velocity_reward = torch.relu(cabinet_dof_vel[:, 3])  # Only reward positive velocity (opening direction)

    # Assign weights to different reward components
    total_reward = 1.0 * transformed_dist_reward + 2.0 * door_opening_reward + 1.5 * door_velocity_reward

    # Normalize total reward to encourage convergence
    total_reward = total_reward / 4.5  # Normalization based on max possible weight sums

    reward_components = {
        "transformed_distance_reward": transformed_dist_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "total_reward": total_reward
    }

    return total_reward, reward_components
