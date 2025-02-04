@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, franka_grasp_rot: torch.Tensor, drawer_grasp_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Reward for minimizing the distance to the drawer
    temp_distance = 2.0  # Increased sensitivity
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Calculate rotational alignment between hand and drawer
    rotation_alignment = torch.sum(franka_grasp_rot * drawer_grasp_rot, dim=-1)

    # Reward for rotational alignment
    temp_rotation = 1.0  # Adjusting sensitivity
    rotation_reward = torch.exp(temp_rotation * rotation_alignment)

    # Reward based on how much the door is opened
    door_opening_extent = cabinet_dof_pos[:, 3]  # Perceiving joint extent as openings
    temp_open = 1.0  # Adjust the scale for opening effect
    open_reward = torch.exp(temp_open * door_opening_extent) - 1.0  # Offset to start from 0

    # Penalty for longer episode lengths to encourage efficiency
    inefficiency_penalty = -0.01

    # Total reward combines distance and rotational alignment with door opening, penalizing inefficiency
    total_reward = distance_reward + rotation_reward + open_reward + inefficiency_penalty

    # Reward components for analysis
    reward_components = {
        "distance_reward": distance_reward,
        "rotation_reward": rotation_reward,
        "open_reward": open_reward,
        "inefficiency_penalty": inefficiency_penalty
    }
    
    return total_reward, reward_components
