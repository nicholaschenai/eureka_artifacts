@torch.jit.script
def compute_reward(
    drawer_pos: torch.Tensor, 
    hand_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor, 
    franka_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Create a device for any new tensors or variables
    device = drawer_pos.device
    
    # Calculate the distance from the hand to the drawer handle
    hand_to_drawer_dist = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Calculate the reward for moving towards the drawer handle
    reward_hand_to_drawer = -hand_to_drawer_dist
    
    # Calculate the reward for opening the cabinet door by checking door position
    # Assume the target goal is something like a specific position value which we can define:
    target_door_pos = torch.tensor([1.0], device=device)  # Hypothetical goal position value

    # Utilize a temperature parameter to encourage exploration by magnifying the reward changes
    temp_door_pos = 0.5
    reward_door_opening = torch.exp(-temp_door_pos * torch.abs(cabinet_dof_pos[:, 3] - target_door_pos))
    
    # Add a reward for the velocity of the door being consistent with opening
    temp_door_vel = 0.5
    reward_door_velocity = torch.exp(-temp_door_vel * torch.abs(cabinet_dof_vel[:, 3]))
    
    # Total Reward
    total_reward = reward_hand_to_drawer + reward_door_opening + reward_door_velocity
    
    # Create a dictionary for individual rewards
    components = {
        "reward_hand_to_drawer": reward_hand_to_drawer,
        "reward_door_opening": reward_door_opening,
        "reward_door_velocity": reward_door_velocity
    }
    
    return total_reward, components
