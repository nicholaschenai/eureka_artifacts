@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Re-written reward for reducing distance to the handle
    # Increase temperature to generate a more favorable gradient
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5
    dist_reward = 1.0 - torch.tanh(distance_to_handle / temperature_distance)

    # Improved reward for opening the door
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_open = 0.5
    opening_restored_reward = torch.exp((door_open_value - 0.2) / temperature_open)  # Encourages more opening

    # Re-scaled the velocity reward to avoid overshadowing other components
    # Focus more on the actions leading to practical outcomes
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 1.0
    velocity_reward = torch.tanh(door_velocity / temperature_velocity)

    # Compute the total reward with balanced weights
    total_reward = 0.5 * dist_reward + 1.5 * opening_restored_reward + 0.3 * velocity_reward

    # Dictionary with individual reward components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
