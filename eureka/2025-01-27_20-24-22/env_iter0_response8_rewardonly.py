@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    drawer_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Ensure that new tensors are on the same device
    device = franka_grasp_pos.device
    
    # Compute the distance between the Franka's hand and the drawer handle
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=1)
    
    # Distance reward
    distance_reward_temp = 0.1
    distance_reward = -torch.exp(-distance_reward_temp * distance_to_drawer)
    
    # Drawer opening reward, assuming opening direction is positive for the given DOF position
    drawer_opening_reward_temp = 0.1
    drawer_opening_reward = torch.exp(drawer_opening_reward_temp * cabinet_dof_pos[:, 3])
    
    # Alignment reward (encouraging hand alignment with the drawer for a full grasp and pull)
    alignment_reward_temp = 0.1
    alignment_vector = drawer_grasp_pos - franka_grasp_pos
    alignment_reward = -torch.exp(-alignment_reward_temp * torch.norm(alignment_vector, dim=1))
    
    # Total reward
    total_reward = distance_reward + drawer_opening_reward + alignment_reward
    
    # Information to return for reward components
    reward_dict = {
        "distance_reward": distance_reward,
        "drawer_opening_reward": drawer_opening_reward,
        "alignment_reward": alignment_reward
    }
    
    return total_reward, reward_dict
