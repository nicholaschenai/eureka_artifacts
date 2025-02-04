@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Redefining the distance reward with adjusted temperature scaling
    temp_distance = 2.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Introducing a velocity-based reward for opening the cabinet
    # Assumes positive velocity indicates opening
    temp_velocity = 0.1
    velocity_reward = torch.exp(temp_velocity * cabinet_dof_vel[:, 3])  # considering only one dof velocity related to the door

    # Penalizing excessive hand velocities to encourage stable approaches
    temp_penalize_velocity = 0.01
    franka_velocity_penalty = torch.exp(-temp_penalize_velocity * torch.norm(franka_grasp_pos, dim=-1))
    
    # Total reward is a combination of being close to the drawer, opening drawer, and stability of hand movement
    total_reward = distance_reward + velocity_reward + franka_velocity_penalty
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "velocity_reward": velocity_reward,
        "franka_velocity_penalty": franka_velocity_penalty
    }
    
    return total_reward, reward_components
