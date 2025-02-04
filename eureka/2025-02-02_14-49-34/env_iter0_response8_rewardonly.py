@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_upper_limits: torch.Tensor,
                   franka_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Calculate distance from hand to drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Reward for reducing the distance to the drawer
    distance_reward_temperature = 1.0
    distance_reward = torch.exp(-distance_to_drawer / distance_reward_temperature)

    # Drawer opening progress (assuming the 3rd DOF corresponds to the drawer)
    drawer_opening_progress = cabinet_dof_pos[:, 3] / cabinet_dof_upper_limits[:, 3]

    # Reward for progress towards opening the drawer
    progress_reward_temperature = 10.0
    progress_reward = torch.exp(drawer_opening_progress / progress_reward_temperature)

    # Penalty for excessive movement
    velocity_penalty_temperature = 0.1
    velocity_penalty = torch.exp(-torch.norm(franka_dof_vel, dim=-1) / velocity_penalty_temperature)

    # Total reward
    total_reward = distance_reward + progress_reward - velocity_penalty

    # Clip the reward to avoid negative values which could confuse the learning process
    total_reward = torch.clamp(total_reward, min=0.0)

    reward_components = {
        "distance_reward": distance_reward,
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
    }

    return total_reward, reward_components
