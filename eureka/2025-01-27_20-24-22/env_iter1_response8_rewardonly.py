@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Adjusted the distance reward to provide a stronger gradient 
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 1.0
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)
    
    # Re-scale the door opening reward to avoid overwhelming other components
    door_opening_reward_temperature = 10.0
    door_opening_scaled = (cabinet_dof_pos[:, 3] / torch.max(cabinet_dof_pos[:, 3])).clamp(min=0, max=1)
    door_opening_reward = torch.exp(door_opening_scaled / door_opening_reward_temperature)

    # Encourage combined small distance and large door opening
    total_reward = 0.5 * distance_reward + 0.5 * door_opening_reward

    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward
    }
    
    return total_reward, reward_dict
