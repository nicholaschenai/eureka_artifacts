@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    torso_quat = root_states[:, 3:7]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Adjusted Forward Velocity Reward
    forward_vel_temperature = 0.15  # Lowered to allow other indicators 
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # New Yaw Control Reward: encourages keeping yaw angle within reasonable limits
    yaw, _, _ = torch.atan2(2.0 * (torso_quat[:, [3]] * torso_quat[:, [2]] + torso_quat[:, [0]] * torso_quat[:, [1]]), 1.0 - 2.0 * (torso_quat[:, [1]]**2 + torso_quat[:, [2]]**2))
    yaw_control_temperature = 0.1
    yaw_reward = -torch.abs(yaw)  # Encourage minimal yaw displacement
    yaw_reward_transformed = torch.exp(yaw_control_temperature * yaw_reward)

    # Adjusted Stability Reward
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.15  # Increased for more differentiation
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))
    
    # Total reward with new weightings
    total_reward = 0.7 * forward_vel_reward + 0.3 * yaw_reward_transformed + 0.3 * stability_reward
    
    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "yaw_reward": yaw_reward_transformed,
        "stability_reward": stability_reward
    }
    
    return total_reward, reward_dict
