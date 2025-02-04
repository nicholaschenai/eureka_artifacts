@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor,
    franka_rfinger_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Temperature parameters for reward transformation
    distance_temp = 1.0
    open_temp = 1.0
    grasp_temp = 1.0

    # Calculate the distance between the hand and the drawer handle
    hand_drawer_distance = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    hand_drawer_reward = torch.exp(-distance_temp * hand_drawer_distance)

    # Encourage drawer to open by measuring the dof position which controls drawer opening
    drawer_opening_reward = torch.exp(open_temp * cabinet_dof_pos[:, 3])

    # Encourage stable grasp by measuring distance between fingers
    finger_distance = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim=-1)
    stable_grasp_reward = torch.exp(-grasp_temp * finger_distance)

    # Compose the total reward
    total_reward = hand_drawer_reward + drawer_opening_reward + stable_grasp_reward

    # Return the total reward and individual components
    reward_components = {
        "hand_drawer_reward": hand_drawer_reward,
        "drawer_opening_reward": drawer_opening_reward,
        "stable_grasp_reward": stable_grasp_reward
    }

    return total_reward, reward_components
