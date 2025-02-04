@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    improved_distance_temperature = 0.05  # Lower temperature to increase reward sensitivity
    distance_reward = torch.exp(-distance_to_handle / improved_distance_temperature)

    # Rescaled Door Opening Reward
    door_opening_factor = torch.clamp(cabinet_dof_pos[:, 3], min=0.0)  # Ensure positive opening values
    improved_door_opening_temperature = 0.5  # Scale down to match distance rewards
    door_opening_reward = torch.exp(door_opening_factor / improved_door_opening_temperature)

    # Total reward as a weighted sum
    total_reward = 0.5 * distance_reward + 0.5 * door_opening_reward  # Equal weight for balanced optimization

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward
    }
    return total_reward, reward_dict
