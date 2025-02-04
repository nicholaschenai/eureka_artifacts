@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Transform distance reward to encourage hand to stay near
    temp_distance = 0.5
    distance_reward = 1.0 - torch.exp(-temp_distance * distance_to_drawer)
   
    # Reward for opening the cabinet door
    # More sensitive transformation to capture small angle changes
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.5  # Increase temperature for more sensitivity
    open_reward = torch.exp(temp_opening * (door_opening_deg / 90.0))

    # Improved speed reward: Encourage continuous motion towards opening
    speed_target = torch.full(cabinet_dof_vel[:, 3].size(), 0.2, device=cabinet_dof_vel.device)
    temp_speed = 1.0
    speed_reward = torch.exp(-temp_speed * torch.abs(cabinet_dof_vel[:, 3] - speed_target))
    speed_reward = torch.where(cabinet_dof_vel[:, 3] > 0.0, speed_reward, torch.zeros_like(speed_reward))

    # Terminal bonus for fully opening the door
    terminal_bonus = torch.where(door_opening_deg >= 90.0, torch.full_like(door_opening_deg, 1.0), torch.zeros_like(door_opening_deg))
    
    # Combine the rewards into a total, weighted sum
    weight_distance = 0.2
    weight_open = 0.5
    weight_speed = 0.2
    weight_terminal = 0.1
    total_reward = (weight_distance * distance_reward + 
                    weight_open * open_reward + 
                    weight_speed * speed_reward + 
                    weight_terminal * terminal_bonus)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "terminal_bonus": terminal_bonus
    }
    
    return total_reward, reward_components
