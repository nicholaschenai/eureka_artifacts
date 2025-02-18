@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    franka_grasp_rot: torch.Tensor,
    drawer_grasp_rot: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance between the gripper's grasp position and drawer's grasp position
    to_drawer_distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    
    # Reward for minimizing the distance to the drawer
    distance_temperature: float = 1.0
    distance_reward = torch.exp(-distance_temperature * to_drawer_distance)
    
    # Calculate drawer opening distance (assuming the fully closed position is represented by cabinet_dof_pos being zero)
    drawer_opening = cabinet_dof_pos[:, 3]

    # Reward for opening the drawer
    opening_temperature: float = 0.5
    opening_reward = torch.exp(opening_temperature * drawer_opening)

    # Calculate alignment of the franka's grasp rotation with the drawer's rotation
    rot_diff = torch.abs(1.0 - torch.bmm(franka_grasp_rot.view(-1, 1, 4), drawer_grasp_rot.view(-1, 4, 1)).squeeze())
    rotation_alignment_reward = torch.exp(-rot_diff)
    
    # Total reward is a weighted sum of the components
    total_reward = distance_reward + opening_reward + rotation_alignment_reward
    
    return total_reward, {
        "distance_reward": distance_reward,
        "opening_reward": opening_reward,
        "rotation_alignment_reward": rotation_alignment_reward
    }
