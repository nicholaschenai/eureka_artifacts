@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced reward for minimizing the distance between the hand and drawer handle.
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_temperature = 0.1  # Increased sensitivity to shorter distances
    dist_reward = torch.exp(-distance_to_handle / distance_temperature) * 2.0

    # Reward for opening the door, adjusted for more balanced contribution
    opening_reward = cabinet_dof_pos[:, 3]
    opening_temperature = 0.2
    opening_restored_reward = torch.exp(opening_reward / opening_temperature) - 1.0

    # Reward for velocity, with a positive-only emphasis and increased scaling
    velocity_reward = torch.clamp(cabinet_dof_vel[:, 3], min=0.0) * 3.0

    # Combine all reward components
    total_reward = 0.7 * dist_reward + 0.5 * opening_restored_reward + 0.3 * velocity_reward

    # Track each individual component for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
