@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance between Franka's grasp position and the drawer's grasp position
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Encourage Franka's hand to reach and grasp the drawer
    reach_reward_temp = 10.0  # Temperature parameter for reach reward
    reach_reward = torch.exp(-reach_reward_temp * distance_to_drawer)

    # Reward for opening the cabinet door
    door_opening_reward_temp = 2.0  # Temperature parameter for the door opening reward
    door_opening_reward = torch.exp(door_opening_reward_temp * cabinet_dof_pos[:, 3])

    # Compute total reward
    total_reward = reach_reward + door_opening_reward

    # Return total reward and individual components
    return total_reward, {"reach_reward": reach_reward, "door_opening_reward": door_opening_reward}
