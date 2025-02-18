@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusting distance reward to encourage precise positioning
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temp_distance = 0.1  # Adjusted for more aggressive scaling
    dist_reward = torch.exp(-distance_to_handle / temp_distance)

    # Reward for door opening
    door_opening_reward = torch.max(-torch.log(-cabinet_dof_pos[:, 3] + 1.0), torch.tensor(0.0))

    # Calibration for movement dynamic exploration
    vel_temperature = 0.2 
    vel_reward = torch.exp(cabinet_dof_vel[:, 3]) * vel_temperature

    # Larger weighted sum that prioritizes tangible progress towards opening
    total_reward = 1.0 * dist_reward + 2.0 * door_opening_reward + 0.5 * vel_reward

    # Map evaluation of reward components
    reward_components = {
        "dist_reward": dist_reward,
        "door_opening_reward": door_opening_reward,
        "vel_reward": vel_reward
    }

    return total_reward, reward_components
