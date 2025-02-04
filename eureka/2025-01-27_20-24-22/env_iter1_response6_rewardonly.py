@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   franka_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted distance reward to improve sensitivity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaling door opening reward for better balance
    door_opening_reward_temperature = 1.0
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Introducing a stability reward to favor smoother actions/reductions in velocity
    stability_reward = torch.mean(torch.abs(franka_dof_vel), dim=-1)
    stability_reward_temperature = 0.1
    stability_reward = torch.exp(-stability_reward / stability_reward_temperature)
    
    # Total reward calculation with balanced components
    total_reward = 0.5 * distance_reward + 0.4 * door_opening_reward + 0.1 * stability_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'stability_reward': stability_reward
    }
    return total_reward, reward_dict
