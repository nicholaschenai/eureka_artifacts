@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Distance reward component: Encourages the end effector to stay close to the drawer handle
    distance_to_handle = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    distance_reward = -distance_to_handle
    
    # Transform the distance reward for stability and normalization
    distance_temp = 0.1
    distance_reward_transformed = 1.0 - torch.exp(-distance_reward / distance_temp)

    # Cabinet openness reward component: Rewards increase in the cabinet's dof position
    # Assuming index 3 reflects the relevant rotational joint for opening
    cabinet_openness = cabinet_dof_pos[:, 3]
    openness_reward = cabinet_openness
    
    # Transform the openness reward for stability and normalization
    openness_temp = 0.1
    openness_reward_transformed = torch.exp(openness_reward / openness_temp) - 1.0

    # Total reward
    total_reward = distance_reward_transformed + openness_reward_transformed

    # Return the rewards
    reward_components = {
        "distance_reward_transformed": distance_reward_transformed,
        "openness_reward_transformed": openness_reward_transformed
    }

    return total_reward, reward_components
