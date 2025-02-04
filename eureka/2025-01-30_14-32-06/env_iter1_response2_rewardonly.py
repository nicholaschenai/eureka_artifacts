@torch.jit.script
def compute_improved_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Reward for forward velocity
    forward_vel_reward = forward_vel

    # Enhanced upright reward
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_reward = torch.sum(up_vec * up_vector_expected, dim=-1)

    # Stability reward based on angular velocity or deviation from upright posture (not implemented in the previous function)
    stability_reward = upright_reward  # Use upright projection as a simple stability indicator

    # Normalizing and transforming the reward components
    forward_vel_temperature = 0.08  # Adjusted temperature for forward velocity
    upright_temperature = 0.3       # Increased temperature for upright posture to strengthen its impact
    stability_temperature = 0.1      # Temperature for the stability component

    forward_vel_reward_transformed = torch.exp(forward_vel_temperature * forward_vel_reward) - 1.0
    upright_reward_transformed = torch.exp(upright_temperature * upright_reward) - 1.0
    stability_reward_transformed = torch.exp(stability_temperature * stability_reward) - 1.0

    # Total reward with adjusted weights
    reward = 0.7 * forward_vel_reward_transformed + 0.2 * upright_reward_transformed + 0.1 * stability_reward_transformed

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return reward, reward_dict
