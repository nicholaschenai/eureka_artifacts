@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, max_episode_length: float, current_episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance to drawer and apply a temperature-transformed reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.5  # Adjusted for better scaling
    distance_reward = 1.0 / (1.0 + temp_distance * distance_to_drawer)
    
    # Adjust open reward using a sigmoid function to provide a gradual increase as the door opens wider
    temp_open = 0.1  # Adjusted for exploration
    open_reward = torch.sigmoid(temp_open * (cabinet_dof_pos[:, 3] - 0.5))  # Assuming a scale where 0.5 is partially open

    # Penalty for longer episode lengths to promote efficient task completion
    temp_length_penalty = 0.01
    length_penalty = -temp_length_penalty * (current_episode_length / max_episode_length)

    # Total reward is composed of both the rewards and the length penalty
    total_reward = distance_reward + open_reward + length_penalty

    # Create the reward components dictionary for debugging purposes
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "length_penalty": length_penalty
    }

    return total_reward, reward_components
