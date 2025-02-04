@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    u_scale: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance to the target drawer position
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Reward for reducing the distance to the target
    distance_reward_temperature = 0.1
    distance_reward = torch.exp(-distance_to_drawer * distance_reward_temperature)

    # Reward for moving the drawer
    movement_reward_temperature = 0.01
    movement_reward = torch.exp(self.cabinet_dof_vel[:, 3] * movement_reward_temperature)

    # Opening reward based on the position of the drawer (assuming the goal is to increase the position value)
    opening_reward = torch.clamp(cabinet_dof_pos[:, 3] * u_scale, min=0.0)

    # Total reward aggregation
    total_reward = distance_reward + movement_reward + opening_reward

    # Constructing the dictionary for each component
    reward_dict = {
        "distance_reward": distance_reward,
        "movement_reward": movement_reward,
        "opening_reward": opening_reward
    }
    
    return total_reward, reward_dict
