@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Forward velocity reward
    forward_vel_temperature = 0.15  # Fine-tune for smoother ascent
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Simplify Upright Reward since it is ineffective
    upright_reward = torch.tensor(0.0, device=root_states.device)

    # Adjust stability reward scaling
    ang_velocity = root_states[:, 10:13]
    stability_weight = 1.0  # Adjust to maintain range, increase if necessary
    stability_reward = stability_weight / (1.0 + torch.norm(ang_velocity, p=2, dim=-1))

    # Compute the total reward
    total_reward = 1.0 * forward_vel_reward + 0.0 * upright_reward + 0.25 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
