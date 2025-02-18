@torch.jit.script
def compute_reward(drawer_grasp_pos: torch.Tensor, franka_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward scaling
    distance_temp: float = 1.0
    velocity_temp: float = 1.0
    target_temp: float = 1.0

    # Compute distance between the gripper and the drawer grasp position
    distance = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Compute the reward for minimizing the distance to the drawer
    distance_reward = torch.exp(-distance_temp * distance)

    # Reward for increase in cabinet DOF position (indicating the door is opening)
    target_reward = torch.exp(target_temp * cabinet_dof_pos[:, 3])

    # Penalize high cabinet door velocity to encourage smooth opening
    velocity_penalty = -torch.exp(velocity_temp * torch.abs(cabinet_dof_vel[:, 3]))

    # Total reward
    total_reward = distance_reward + target_reward + velocity_penalty

    # Pack individual rewards into a dictionary for diagnostics
    reward_components = {
        "distance_reward": distance_reward,
        "target_reward": target_reward,
        "velocity_penalty": velocity_penalty
    }

    return total_reward, reward_components
