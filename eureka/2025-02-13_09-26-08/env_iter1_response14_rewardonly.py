@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Step 1: Reduce the distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = -distance_to_handle  # Negative of distance encourages reaching the drawer handle
    temperature_distance = 0.1
    transformed_distance_reward = torch.exp(distance_reward / temperature_distance)

    # Step 2: Motivate literal opening of the door
    # We assume a positive change in door position means opening
    door_opening_position = cabinet_dof_pos[:, 3]
    door_position_reward = torch.relu(door_opening_position)  # Reward based on how open the door is
    
    # Step 3: Encourage velocity in the opening direction
    door_velocity = cabinet_dof_vel[:, 3]
    door_velocity_reward = torch.relu(door_velocity)  # Positive velocity towards opening should be rewarded

    # Recalculate total reward with appropriate scaling
    total_reward = 1.0 * transformed_distance_reward + 2.0 * door_position_reward + 1.0 * door_velocity_reward

    # Create components dict for analysis
    reward_components = {
        "distance_to_handle": distance_reward,
        "transformed_distance_reward": transformed_distance_reward,
        "door_position_reward": door_position_reward,
        "door_velocity_reward": door_velocity_reward
    }

    return total_reward, reward_components
