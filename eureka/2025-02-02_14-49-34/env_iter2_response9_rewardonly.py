@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 2.0  # Increased sensitivity
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # Encourages minimal distance with exponential decay

    # Open reward
    door_angle_rad = cabinet_dof_pos[:, 3]
    door_open_pos = torch.clamp(door_angle_rad, 0.0, 1.57)  # Ensure radian range for 0 to 90 degrees
    temp_open = 0.5  # Adjust sensitivity to normalization
    open_reward = torch.exp(temp_open * door_open_pos) - 1  # Reward increased opening positions

    # Total reward - encourage more towards opening (increase weight)
    weight_distance = 0.3
    weight_open = 0.7
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }

    return total_reward, reward_components
