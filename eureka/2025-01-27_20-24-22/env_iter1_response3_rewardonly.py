@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Re-worked reward for the distance to the handle of the door
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 2.0  # Increased temp to give more gradient
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Re-worked door opening reward
    door_opening_value = cabinet_dof_pos[:, 3]
    # Reward component re-scaled to be more comparable with the distance reward
    door_opening_reward_temperature = 10.0  # Increased to make it less dominating
    door_opening_reward = torch.exp(door_opening_value / door_opening_reward_temperature)

    # Normalize the rewards to prevent dominance of one component
    normalized_distance_reward = distance_reward / (distance_reward + door_opening_reward)
    normalized_door_opening_reward = door_opening_reward / (distance_reward + door_opening_reward)

    # Incentivize a decrease in episode lengths (i.e., a faster task completion)
    length_penalty_temperature = 1.0
    length_penalty = torch.exp(-cabinet_dof_pos[:, 3] / length_penalty_temperature)

    # Total reward with equal weight for balanced learning
    total_reward = normalized_distance_reward + normalized_door_opening_reward + 0.1 * length_penalty

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'normalized_distance_reward': normalized_distance_reward,
        'normalized_door_opening_reward': normalized_door_opening_reward,
        'length_penalty': length_penalty
    }
    return total_reward, reward_dict
