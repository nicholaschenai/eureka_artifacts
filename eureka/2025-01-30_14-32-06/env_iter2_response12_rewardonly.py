@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance to target and expected direction
    torso_position = root_states[:, 0:3]
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)

    # Revised Forward Velocity Reward
    forward_vel_temperature = 0.1  # Adjusted to increase sensitivity
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Revised Upright Reward
    # Make the upright orientation more sensitive to changes away from the ideal up vector
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 2.0  # Increased to enhance sensitivity
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 1.0))

    # Combine the rewards
    total_reward = 1.5 * forward_vel_reward + 0.7 * upright_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward
    }

    return total_reward, reward_dict
