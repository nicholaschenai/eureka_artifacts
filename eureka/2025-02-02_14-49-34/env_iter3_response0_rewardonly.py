@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Inverted distance reward to encourage hand to be near the drawer
    temp_distance = 0.3
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)

    # Reward for opening the cabinet door
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.5
    open_reward = torch.tanh(temp_opening * torch.sin(torch.deg2rad(door_opening_deg)))

    # Total reward as a weighted sum of components
    weight_distance = 0.4
    weight_open = 0.6
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Clamp total reward to ensure it remains non-negative
    total_reward = torch.clamp(total_reward, min=0.0)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }

    return total_reward, reward_components
