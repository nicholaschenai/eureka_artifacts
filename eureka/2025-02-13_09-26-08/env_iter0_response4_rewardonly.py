@torch.jit.script
def compute_reward(franka_grasp_pos: Tensor, drawer_grasp_pos: Tensor, cabinet_dof_pos: Tensor, cabinet_dof_vel: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    # Compute distance to target (drawer handle)
    distance_to_target = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward = 1.0 / (1.0 + distance_to_target)  # A higher reward closer to target
    distance_temperature = 0.1
    distance_reward = torch.exp(-distance_temperature * distance_to_target)

    # Reward for opening the cabinet drawer
    open_reward = cabinet_dof_pos[:, 3]  # Assuming the 3rd index corresponds to the drawer's opened position
    open_temperature = 0.05
    open_reward = torch.exp(open_temperature * open_reward)

    # Penalize high velocity
    velocity_penalty = cabinet_dof_vel[:, 3] ** 2
    velocity_temperature = 0.01
    velocity_penalty = torch.exp(-velocity_temperature * velocity_penalty)

    # Total reward
    total_reward = distance_reward + open_reward - velocity_penalty

    # Components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "velocity_penalty": -velocity_penalty
    }
    
    return total_reward, reward_components
