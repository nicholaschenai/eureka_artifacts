@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Rescaled distance reward
    temp_distance = 0.4
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
   
    # Enhanced reward for opening the cabinet door
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.2  # Scaled to be more sensitive
    open_reward = torch.tanh(temp_opening * door_opening_deg)
    
    # Rewritten component: Reward for door movement directionality
    # Encouraging consistent positive velocity towards opening
    temp_velocity = 0.5
    positive_velocity = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    speed_reward = torch.tanh(temp_velocity * positive_velocity)

    # Reward for successful task completion (e.g., fully opened door)
    # High reward if the door opens beyond a certain degree threshold
    successful_completion_threshold = 70.0  # Degrees for fully opened door
    completion_reward = torch.where(door_opening_deg >= successful_completion_threshold, torch.tensor(1.0, device=door_opening_deg.device), torch.tensor(0.0, device=door_opening_deg.device))

    # Combine the rewards into a total, weighted sum
    weight_distance = 0.2
    weight_open = 0.4
    weight_speed = 0.1
    weight_completion = 0.3
    total_reward = (weight_distance * distance_reward 
                    + weight_open * open_reward 
                    + weight_speed * speed_reward 
                    + weight_completion * completion_reward)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "completion_reward": completion_reward
    }
    
    return total_reward, reward_components
