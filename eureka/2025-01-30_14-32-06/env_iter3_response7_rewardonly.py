@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    distance_to_target = torch.norm(to_target, p=2, dim=-1)
    forward_vel = torch.sum(velocity * torch.nn.functional.normalize(to_target, dim=-1), dim=-1)
    
    # Adjusted Reward for forward velocity
    forward_vel_temperature = 0.1  # Slight increase to maintain competitiveness
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Upright Reward - Encourage posture change by relating it to movement
    movement = torch.norm(velocity, p=2, dim=-1)
    upright_temperature = 0.5
    upright_reward = torch.exp(upright_temperature * (movement * (up_vec[:, 2] - 0.5)))

    # Stability Reward - Address angular velocity rather than position
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.05
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Add Distance Reward to encourage reaching the target
    distance_temperature = 0.1
    distance_reward = torch.exp(-distance_temperature * distance_to_target)

    # Total reward is a weighted sum of the components
    total_reward = 1.0 * forward_vel_reward + 0.2 * upright_reward + 0.2 * stability_reward + 0.5 * distance_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward,
        "distance_reward": distance_reward
    }

    return total_reward, reward_dict
