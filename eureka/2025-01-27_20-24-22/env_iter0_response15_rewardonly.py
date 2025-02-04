@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    franka_grasp_rot: torch.Tensor,
    drawer_grasp_rot: torch.Tensor,
    franka_dof_vel: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Reward components
    drawer_opening_pos = cabinet_dof_pos[:, 3]  # Assuming this represents the drawer position
    
    # 1. Drawer Opening Position Reward
    drawer_open_reward = drawer_opening_pos
    drawer_opening_temp = 0.1
    drawer_open_reward_transformed = torch.exp(drawer_opening_temp * drawer_open_reward)
    
    # 2. Distance to Drawer Reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, p=2, dim=1)
    distance_to_drawer_temp = -1.0
    distance_to_drawer_reward = torch.exp(distance_to_drawer_temp * distance_to_drawer)
    
    # 3. Alignment Reward
    alignment_diff = torch.norm(drawer_grasp_rot - franka_grasp_rot, p=2, dim=1)
    alignment_temp = -0.5
    alignment_reward = torch.exp(alignment_temp * alignment_diff)
    
    # 4. Smoothness Penalty
    smoothness_penalty = torch.norm(franka_dof_vel, p=2, dim=1)
    smoothness_temp = -0.01
    smoothness_penalty_transformed = torch.exp(smoothness_temp * smoothness_penalty)
    
    # Total reward
    total_reward = (
        drawer_open_reward_transformed +
        distance_to_drawer_reward +
        alignment_reward -
        smoothness_penalty_transformed
    )
    
    # Individual components for debugging/analysis
    reward_components = {
        "drawer_open_reward": drawer_open_reward_transformed,
        "distance_to_drawer_reward": distance_to_drawer_reward,
        "alignment_reward": alignment_reward,
        "smoothness_penalty": -smoothness_penalty_transformed
    }
    
    return total_reward, reward_components
