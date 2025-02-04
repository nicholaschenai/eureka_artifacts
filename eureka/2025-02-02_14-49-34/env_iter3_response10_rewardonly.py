@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Normalized and inverted distance reward to encourage hand to be near the drawer
    temp_distance = 0.25
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)
    
    # Reward for opening the cabinet door
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.2  # Making it more sensitive
    open_reward = torch.tanh(temp_opening * door_opening_deg)

    # Adjust speed reward: Encourage consistent positive velocity
    temp_velocity = 0.3
    speed_reward = torch.clip(cabinet_dof_vel[:, 3], min=0.0)
    speed_reward = torch.tanh(temp_velocity * speed_reward)

    # Add penalty for large distances as a deterrent
    temp_penalty_dist = 0.3
    distance_penalty = torch.tanh(temp_penalty_dist * distance_to_drawer)

    # Combine the rewards and balance them
    total_reward = 0.3 * distance_reward + 0.4 * open_reward + 0.3 * speed_reward - 0.1 * distance_penalty
    total_reward = torch.clamp(total_reward, min=0.0)

    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "distance_penalty": -distance_penalty
    }

    return total_reward, reward_components
