@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for distance to the handle of the door
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.1
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Reward for opening the door, assumed to be the 4th dof in cabinet_dof_pos
    door_opening_reward_temperature = 0.1
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Total reward calculation
    total_reward = distance_reward + door_opening_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward
    }
    return total_reward, reward_dict
