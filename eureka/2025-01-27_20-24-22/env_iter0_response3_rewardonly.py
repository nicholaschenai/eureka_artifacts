@torch.jit.script
def compute_reward(hand_pos: torch.Tensor, drawer_pos: torch.Tensor, franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Hyperparameters
    position_temperature = 10.0  # temperature parameter for position distance
    dof_position_temperature = 5.0  # temperature parameter for DOF position
    target_dof_position = 1.0  # assuming this is the fully open position of the cabinet DOF

    # Calculate distance from hand to drawer grasp position
    hand_to_drawer_dist = torch.norm(hand_pos - drawer_grasp_pos, dim=-1)

    # Reward for minimizing hand to drawer distance
    position_reward = torch.exp(-position_temperature * hand_to_drawer_dist)

    # Encourage the cabinet door to reach the fully open position
    dof_reward = torch.exp(-dof_position_temperature * torch.abs(cabinet_dof_pos - target_dof_position))

    # Total reward is a weighted sum of position and DOF rewards
    total_reward = position_reward + dof_reward

    # Return total reward and individual components
    reward_components = {
        'position_reward': position_reward,
        'dof_reward': dof_reward
    }

    return total_reward, reward_components
