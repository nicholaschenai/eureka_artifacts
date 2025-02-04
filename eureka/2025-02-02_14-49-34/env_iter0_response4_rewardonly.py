@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, 
    franka_rfinger_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance-based reward for moving the hand closer to the drawer handle
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = -distance

    # Reward for the cabinet being open, which we'll model as the position of a specific DOF
    cabinet_open_reward = cabinet_dof_pos[:, 3]  # Assuming the specific DOF controlling the cabinet door is at index 3

    # Velocity penalty to encourage steady opening instead of rapid, potentially damaging movement
    velocity_penalty = -0.1 * torch.abs(cabinet_dof_vel[:, 3])

    # Finger alignment reward to ensure a proper grip on the handle
    temperature_alignment = 10.0  # Temperature for alignment reward transformation
    finger_alignment = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim=-1)
    alignment_reward = torch.exp(-finger_alignment * temperature_alignment)

    # Overall reward as a weighted sum of components
    total_reward = distance_reward + cabinet_open_reward + velocity_penalty + alignment_reward

    reward_dict = {
        "distance_reward": distance_reward,
        "cabinet_open_reward": cabinet_open_reward,
        "velocity_penalty": velocity_penalty,
        "alignment_reward": alignment_reward
    }

    return total_reward, reward_dict
