@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Define a stronger distance component to incentivize approaching the handle:
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_dist = 0.1  # Tighter temperature for sharper decline
    dist_reward = torch.exp(-distance_to_handle / temperature_dist)

    # Reframe and boost opening reward with an emphasis on movement instead of static measurement:
    opening_extent = torch.clamp(cabinet_dof_pos[:, 3], min=0.0)
    temperature_open = 1.0
    opening_reward = torch.exp(opening_extent / temperature_open) - 1.0  # Boosted transform

    # Give a scaled boost to the velocity reward:
    velocity_boost = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    velocity_reward = velocity_boost * 0.5  # Half scaling to improve effect

    # Integrate all components, adjusting their weights to ensure balanced learning focus:
    total_reward = 1.5 * dist_reward + 2.0 * opening_reward + 0.5 * velocity_reward

    # Collect each individual component in a dictionary for clear debug insights:
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
