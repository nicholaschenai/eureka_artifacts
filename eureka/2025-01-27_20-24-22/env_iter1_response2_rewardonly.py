@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for reducing distance to the handle of the door
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.05
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)
    distance_reward = distance_reward * 2.0  # Re-scaling to make it more impactful

    # Reward for opening the door, assume related to the 4th dof
    door_opening_reward_temperature = 5.0
    door_opening_norm_reward = torch.sigmoid(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Total reward calculation
    total_reward = distance_reward + door_opening_norm_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_norm_reward': door_opening_norm_reward
    }
    return total_reward, reward_dict
