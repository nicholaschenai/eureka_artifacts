@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, 
                   torso_quat: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute velocity reward which encourages moving towards the target
    forward_velocity_reward = potentials - prev_potentials  # Faster movement to the target yields higher reward
    forward_velocity_reward_temperature = 0.1
    forward_velocity_reward = torch.exp(forward_velocity_reward / forward_velocity_reward_temperature)
    
    # Compute balance reward; we want to keep the torso upright
    up_axis_idx = 2  # assuming Z-axis is up
    balance_reward = torch.abs(torso_quat[:, up_axis_idx] - 1.0)  # reward high when upright
    balance_reward_temperature = 0.5
    balance_reward = torch.exp(-balance_reward / balance_reward_temperature)
    
    # Regularize joint movement to avoid unnecessary motion
    dof_vel_penalty = torch.sum(dof_vel ** 2, dim=-1)  # penalize high joint velocities
    dof_vel_penalty_temperature = 1.0
    dof_vel_penalty = torch.exp(-dof_vel_penalty / dof_vel_penalty_temperature)

    # Combine rewards plus penalty
    total_reward = forward_velocity_reward + balance_reward - dof_vel_penalty
    
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "balance_reward": balance_reward,
        "dof_vel_penalty": -dof_vel_penalty
    }
    
    return total_reward, reward_components
