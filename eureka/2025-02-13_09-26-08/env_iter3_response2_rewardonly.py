@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Distance reward - increase precision in reaching the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Smaller temperature for increased incentive to reduce distance
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Door opening - balance encouraging open and stabilize at open
    door_open_value = torch.clamp(cabinet_dof_pos[:, 3], 0.0, 1.0)
    opening_reward = door_open_value * 3.0  # Linear scaling to encourage more door opening
    
    # Smooth velocity reward to avoid overpowering contributions
    door_velocity = cabinet_dof_vel[:, 3].clamp(min=-1.0, max=1.0)  # Limit velocity impact range
    velocity_reward_scale = 0.2
    velocity_reward = door_velocity * velocity_reward_scale

    # Final reward balancing all components
    total_reward = dist_reward + opening_reward + velocity_reward

    # Compile reward components for inspection
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
