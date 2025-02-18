@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for minimizing the distance between the hand and the target grasp position
    distance_to_grasp = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward_temperature = 0.1
    distance_reward = torch.exp(-distance_reward_temperature * distance_to_grasp)
    
    # Reward for opening the drawer (increasing DOF position value)
    drawer_opening_progress = cabinet_dof_pos[:, 3]
    opening_reward_temperature = 0.5
    opening_reward = torch.exp(opening_reward_temperature * drawer_opening_progress)

    # Penalty for high velocities in the drawer DOF
    velocity_penalty = torch.square(cabinet_dof_vel[:, 3])
    velocity_penalty_temperature = 0.1
    velocity_penalty = torch.exp(-velocity_penalty_temperature * velocity_penalty)

    # Total combined reward
    total_reward = distance_reward + opening_reward - 0.1 * velocity_penalty

    # Dictionary of individual reward components
    reward_components = {
        'distance_reward': distance_reward,
        'opening_reward': opening_reward,
        'velocity_penalty': velocity_penalty
    }

    return total_reward, reward_components
