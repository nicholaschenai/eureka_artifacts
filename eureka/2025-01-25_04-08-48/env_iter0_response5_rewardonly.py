@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, dof_vel: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Normalize the forward velocity reward
    forward_velocity_reward = velocity[:, 0]  # Assuming x-axis is forward
    forward_velocity_reward_temp = 0.1  # Temperature parameter for forward velocity
    forward_velocity_reward = torch.exp(forward_velocity_reward / forward_velocity_reward_temp) - 1.0

    # Penalize high joint velocities
    joint_velocity_penalty = torch.sum(dof_vel ** 2, dim=1)
    joint_velocity_penalty_temp = 0.01  # Temperature parameter for joint velocity penalty
    joint_velocity_penalty = torch.exp(-joint_velocity_penalty / joint_velocity_penalty_temp)

    # Encourage the humanoid to stay upright
    upright_reward = up_vec[:, 2]  # Assuming z-axis is up
    upright_reward_temp = 0.05  # Temperature parameter for upright reward
    upright_reward = torch.exp(upright_reward / upright_reward_temp) - 1.0

    # Combine rewards
    total_reward = forward_velocity_reward + joint_velocity_penalty + upright_reward

    # Collect reward components
    reward_components = {
        'forward_velocity_reward': forward_velocity_reward,
        'joint_velocity_penalty': joint_velocity_penalty,
        'upright_reward': upright_reward
    }

    return total_reward, reward_components
