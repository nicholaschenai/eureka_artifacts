@torch.jit.script
def compute_reward(hand_pos: torch.Tensor, drawer_pos: torch.Tensor, hand_rot: torch.Tensor, drawer_rot: torch.Tensor,
                   franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, franka_dof_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Temperature parameters for reward components
    distance_temp = 10.0
    rotation_temp = 1.0
    opening_temp = 1.0

    # Calculate distance between the gripper and the drawer handle
    hand_to_drawer_distance = torch.norm(hand_pos - drawer_pos, p=2, dim=-1)

    # Calculate the conformity of the orientation between hand and the drawer handle
    hand_to_drawer_rotation_similarity = torch.nn.functional.cosine_similarity(hand_rot, drawer_rot, dim=-1)

    # Calculate finger distance to check if they are at the handle
    left_finger_to_drawer = torch.norm(franka_lfinger_pos - drawer_pos, p=2, dim=-1)
    right_finger_to_drawer = torch.norm(franka_rfinger_pos - drawer_pos, p=2, dim=-1)
    finger_distance = 0.5 * (left_finger_to_drawer + right_finger_to_drawer)

    # Reward for minimizing distance to the drawer handle
    distance_reward = torch.exp(-distance_temp * hand_to_drawer_distance)

    # Reward for proper gripper-hand orientation
    rotation_reward = hand_to_drawer_rotation_similarity

    # Reward for opening the drawer (measured by the cabinet's degree of opening)
    opening_reward = torch.exp(opening_temp * cabinet_dof_pos[:, 3])  # Assuming index 3 is the drawer opening dof

    # Total reward is a weighted sum of individual components
    total_reward = distance_reward + rotation_reward + opening_reward

    # Construct a dictionary to return individual reward components
    reward_components = {
        "distance_reward": distance_reward,
        "rotation_reward": rotation_reward,
        "opening_reward": opening_reward
    }

    return total_reward, reward_components
