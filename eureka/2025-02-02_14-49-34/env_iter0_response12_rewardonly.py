@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    drawer_pos: torch.Tensor,
    initial_drawer_pos: torch.Tensor, 
    franka_dof_pos: torch.Tensor,
    franka_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate the distance the drawer has been pulled out
    pull_distance = torch.norm(drawer_pos - initial_drawer_pos, p=2, dim=-1)
    
    # Reward for opening the drawer
    pull_reward = pull_distance

    # Encourage the robot's end effector to remain close to the drawer handle
    alignment_distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    temperature_alignment = 0.1
    alignment_reward = torch.exp(-temperature_alignment * alignment_distance)
    
    # Penalize the robot for high velocity or unstable movements
    velocity_penalty = torch.sum(franka_dof_vel ** 2, dim=-1)
    stability_reward = -velocity_penalty
    
    # Combine components to form the total reward
    total_reward = pull_reward + alignment_reward + stability_reward
    
    # Normalize total reward
    temperature_total = 0.1
    normalized_total_reward = torch.exp(temperature_total * total_reward) - 1.0

    # Create dictionary of individual reward components
    reward_components = {
        "pull_reward": pull_reward,
        "alignment_reward": alignment_reward,
        "stability_reward": stability_reward
    }
    
    return normalized_total_reward, reward_components
