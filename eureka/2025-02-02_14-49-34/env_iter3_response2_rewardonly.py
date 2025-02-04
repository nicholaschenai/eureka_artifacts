@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Normalized and inverted distance reward to encourage hand to be near the drawer
    temp_distance = 0.5
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
   
    # Reward for opening the cabinet door
    door_opening_angle = cabinet_dof_pos[:, 3]
    temp_opening = 1.0
    open_reward = torch.exp(temp_opening * torch.clamp(door_opening_angle, max=1.0))  # Assume angle is in radians

    # New completion reward to give a significant bonus when fully open
    door_fully_open = (door_opening_angle >= 1.0)  # Assuming 1 radian is the fully opened position
    completion_reward = torch.where(door_fully_open, torch.tensor(2.0, device=franka_grasp_pos.device), torch.tensor(0.0, device=franka_grasp_pos.device))

    # Combine the rewards into a total
    weight_distance = 0.2
    weight_open = 0.5
    weight_completion = 0.3
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion * completion_reward

    # Clamp total reward to ensure it remains non-negative
    total_reward = torch.clamp(total_reward, min=0.0)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_reward": completion_reward
    }
    
    return total_reward, reward_components
