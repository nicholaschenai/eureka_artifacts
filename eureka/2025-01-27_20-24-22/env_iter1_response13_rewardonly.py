@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Revised Reward for distance to the handle of the door
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 1.0  # Adjusted temperature to modulate range
    distance_reward = 1.0 / (1.0 + torch.exp(distance / distance_reward_temperature))

    # Rescaled reward for opening the door
    door_opening_reward_normalization = 0.1  # Downscale factor for the door opening
    door_opening_reward = cabinet_dof_pos[:, 3] * door_opening_reward_normalization

    # Total reward calculation
    total_reward = distance_reward + door_opening_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward
    }
    return total_reward, reward_dict
