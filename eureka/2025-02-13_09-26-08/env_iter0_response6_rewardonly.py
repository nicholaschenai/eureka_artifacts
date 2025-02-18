@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    franka_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance between the drawer grasp position and the robot hand grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Encouragement to minimize the distance
    distance_reward_temperature = 0.1  # temperature parameter for distance reward
    distance_reward = torch.exp(-distance_to_drawer / distance_reward_temperature)

    # Reward for opening the cabinet door
    door_opening_reward_temperature = 0.1  # temperature parameter for door opening reward
    door_opening_reward = torch.exp(cabinet_dof_pos / door_opening_reward_temperature).squeeze(-1)

    # Optionally, penalize for high velocity to encourage stable movement
    velocity_penalty_temperature = 0.01  # temperature parameter for velocity penalty
    velocity_penalty = torch.exp(-torch.norm(franka_dof_vel, dim=-1) / velocity_penalty_temperature)
    
    # Total reward is a combination of all individual rewards
    total_reward = distance_reward + door_opening_reward + 0.1 * velocity_penalty

    # Reward components
    reward_components = {
        "distance_reward": distance_reward,
        "door_opening_reward": door_opening_reward,
        "velocity_penalty": velocity_penalty
    }
    
    return total_reward, reward_components
