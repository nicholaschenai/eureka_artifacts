@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Reward for opening the cabinet door
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.05  # Lowered to stabilize opening rewards
    open_reward = torch.tanh(temp_opening * door_opening_deg)
    
    # Remove the speed reward; it was ineffective in current setting
    # stronger reward for fully opened door
    threshold_open_degrees = torch.tensor(90.0, device=cabinet_dof_pos.device)
    success_bonus = torch.where(door_opening_deg >= threshold_open_degrees, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))

    # Penalize agent if door is not opened after max steps
    penalty = torch.tensor(-1.0, device=cabinet_dof_pos.device) * (cabinet_dof_pos[:, 3] < threshold_open_degrees) 
    
    # Combine the rewards into a total, weighted sum
    weight_distance = 0.3
    weight_open = 0.6
    weight_success_bonus = 0.2

    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_success_bonus * success_bonus + penalty

    # Clamp total reward to ensure it remains non-negative
    total_reward = torch.clamp(total_reward, min=0.0)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "success_bonus": success_bonus
    }
    
    return total_reward, reward_components
