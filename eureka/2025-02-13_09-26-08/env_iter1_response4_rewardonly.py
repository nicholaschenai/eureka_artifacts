@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for minimizing distance to the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1
    transformed_distance_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Create a positive reward for door opening – scale to substantially incentivize movement
    temperature_door_opening = 0.05
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / temperature_door_opening) - 1.0

    # Encourage velocity in opening direction - small penalty for negative velocity
    door_velocity_reward = cabinet_dof_vel[:, 3]
    temperature_door_velocity = 0.01
    transformed_velocity_reward = torch.exp(door_velocity_reward / temperature_door_velocity) - 1.0

    # Total Reward
    total_reward = transformed_distance_reward + 2.0 * door_opening_reward + transformed_velocity_reward

    reward_components = {
        "transformed_distance_reward": transformed_distance_reward,
        "door_opening": door_opening_reward,
        "door_velocity": transformed_velocity_reward
    }

    return total_reward, reward_components
