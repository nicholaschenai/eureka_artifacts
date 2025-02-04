@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Reward close proximity
    # New reward attempt with exponential decay to reinforce closing distance
    temp_distance = 1.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  

    # New open reward refocuses on normalized degree of opening
    max_open_angle = 90.0  # degrees
    temp_opening = 0.1
    current_open = cabinet_dof_pos[:, 3] / torch.deg2rad(torch.tensor(max_open_angle, device=franka_grasp_pos.device))
    open_reward = torch.clamp(current_open, 0.0, 1.0)

    # Penalize for long durations without achieving goals: penalize episode length
    temp_duration = 0.01
    duration_penalty = torch.exp(temp_duration * distance_to_drawer)

    # Total reward is a combination of reinforcement for task efficient closeness and opening, penalizing long episodes
    total_reward = distance_reward + open_reward - duration_penalty

    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "duration_penalty": duration_penalty
    }

    return total_reward, reward_components
