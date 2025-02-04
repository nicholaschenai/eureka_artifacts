@torch.jit.script
def compute_reward(drawer_grasp_pos: torch.Tensor, franka_grasp_pos: torch.Tensor,
                   drawer_rot: torch.Tensor, hand_rot: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Reward for minimizing the distance between the robot hand and the drawer handle
    distance_reward = -torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Reward for aligning the hand with the drawer for effective grasping
    alignment_reward = torch.sum(hand_rot * drawer_rot, dim=-1)  # cosine similarity for orientation
    
    # Reward for the actual opening of the drawer (e.g., maximizing the cabinet door's rotation or translation position)
    door_opening_reward = cabinet_dof_pos[:, 3]  # Assuming the 4th DoF corresponds to the door opening

    # Transformations and normalization
    temperature_distance = 1.0
    temperature_alignment = 1.0
    temperature_door_opening = 1.0

    transformed_distance_reward = torch.exp(distance_reward / temperature_distance)
    transformed_alignment_reward = torch.exp(alignment_reward / temperature_alignment)
    transformed_door_opening_reward = torch.exp(door_opening_reward / temperature_door_opening)

    # Combining all rewards into a single scalar value
    total_reward = transformed_distance_reward + transformed_alignment_reward + transformed_door_opening_reward

    # Collecting each reward component for detailed analysis
    reward_components = {
        "distance_reward": transformed_distance_reward,
        "alignment_reward": transformed_alignment_reward,
        "door_opening_reward": transformed_door_opening_reward
    }

    return total_reward, reward_components
