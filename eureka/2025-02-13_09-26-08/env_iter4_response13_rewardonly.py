@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced distance reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 10.0
    
    # Re-scaled opening reward
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 0.5
    opening_restored_reward = torch.exp(torch.abs(door_open_value) / temperature_opening) * 1.0
    
    # Revised velocity reward to encourage smoother movement
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.5
    velocity_reward = 1.0 / (1.0 + torch.exp(-door_velocity / temperature_velocity))

    # Composite total reward
    total_reward = dist_reward + opening_restored_reward + velocity_reward

    # Dictionary with individual components
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
