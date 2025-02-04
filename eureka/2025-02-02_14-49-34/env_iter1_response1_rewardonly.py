@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Reward agent for minimizing the distance
    temp_distance = 0.5  # Adjusted temperature for finer scaling
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)

    # Reward dynamic opening of the cabinet door
    target_dof_pos = 1.0  # Assuming max open position is 1.0
    temp_open = 2.0  # Adjust temperature for more significant reward at progress
    open_reward = torch.exp(temp_open * (cabinet_dof_pos[:, 3] - target_dof_pos))

    # Penalize longer episode durations to encourage quick completion
    max_length = 500.0
    temp_length = 0.01
    time_penalty = torch.exp(temp_length * (-episode_length / max_length))

    # Total reward is a combination of being close to the drawer, opening the door, and completing the task quickly
    total_reward = distance_reward + open_reward + time_penalty

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "time_penalty": time_penalty
    }

    return total_reward, reward_components
