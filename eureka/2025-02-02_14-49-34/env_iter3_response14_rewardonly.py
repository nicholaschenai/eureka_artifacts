@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Scale down slightly to avoid overpowering other components
    temp_distance = 0.3
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Reward for opening the cabinet door with stronger scaling
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.3
    open_reward = torch.tanh(temp_opening * door_opening_deg)

    # Enhancing the speed reward to better measure effectiveness
    temp_velocity = 1.0
    speed_reward = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)  # Focusing on positive velocities only
    speed_reward = torch.tanh(temp_velocity * speed_reward)

    # Adding a new component: Reward for achieving a certain opening target
    opening_target_deg = 80.0  # Assume target opening degree; adjustable based on task requirements
    target_open_reward = torch.where(door_opening_deg > opening_target_deg, torch.ones_like(door_opening_deg), torch.zeros_like(door_opening_deg))

    # Combine the rewards into a total, weighted sum
    weight_distance = 0.2
    weight_open = 0.5
    weight_speed = 0.1
    weight_target_open = 0.2
    total_reward = (weight_distance * distance_reward + weight_open * open_reward +
                    weight_speed * speed_reward + weight_target_open * target_open_reward)

    # Clamp total reward to ensure it remains non-negative
    total_reward = torch.clamp(total_reward, min=0.0)
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "target_open_reward": target_open_reward
    }
    
    return total_reward, reward_components
