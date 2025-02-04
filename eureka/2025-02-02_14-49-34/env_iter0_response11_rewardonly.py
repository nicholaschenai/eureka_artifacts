@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance between the gripper and the drawer
    grasp_dist = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Reward for minimizing distance, use exponential scaling to encourage closer proximity
    temperature_dist = 5.0  # Temperature parameter for grasp distance reward
    reward_grasp_dist = torch.exp(-temperature_dist * grasp_dist)

    # Reward for opening the cabinet door (assuming full open position is represented by highest value within the range)
    target_drawer_opening = 1.0  # Assume target at the maximum normalized position
    dof_open_reward = torch.relu(cabinet_dof_pos[:, 3] - target_drawer_opening)
    
    # Temperature parameter for opening reward
    temperature_opening = 2.0
    reward_opening = torch.exp(-temperature_opening * dof_open_reward)

    # Combine rewards
    total_reward = reward_grasp_dist + reward_opening

    return total_reward, {
        'reward_grasp_dist': reward_grasp_dist,
        'reward_opening': reward_opening,
    }
