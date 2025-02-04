@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance to the drawer and offer penalty for distance
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    max_distance = 2.0  # Hypothetical maximum distance the agent can be from the drawer
    temp_distance = 2.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer / max_distance)  # Penalize being further away

    # Reward for opening the door fully
    door_angle_deg = torch.rad2deg(cabinet_dof_pos[:, 3])  # Assuming the third degree of freedom controls the door
    target_open_deg = 90.0  # Hypothetical goal for full door open angle
    temp_open = 0.1
    open_reward = torch.exp(-temp_open * torch.abs(target_open_deg - door_angle_deg) / target_open_deg)  # Reward reaching desired open angle

    # Encourage simultaneous minimization of distance while maximizing the door open angle
    weight_distance = 0.4
    weight_open = 0.6
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Creating the reward components dictionary for better insight during training
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
