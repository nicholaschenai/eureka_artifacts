@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor,
                   franka_lfinger_pos: torch.Tensor,
                   franka_rfinger_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Updated Reward for Distance to the handle of the door
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 1.0 
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Scaled Reward for opening the door
    door_opening_reward_temperature = 5.0
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # New Reward for grip stability (ensures that both fingers are close to the target)
    grip_stability = (torch.norm(franka_lfinger_pos - drawer_grasp_pos, p=2, dim=-1) +
                      torch.norm(franka_rfinger_pos - drawer_grasp_pos, p=2, dim=-1)) * 0.5
    grip_stability_temperature = 0.5
    grip_stability_reward = torch.exp(-grip_stability / grip_stability_temperature)

    # Total reward calculation
    total_reward = distance_reward + door_opening_reward + grip_stability_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'grip_stability_reward': grip_stability_reward
    }
    return total_reward, reward_dict
