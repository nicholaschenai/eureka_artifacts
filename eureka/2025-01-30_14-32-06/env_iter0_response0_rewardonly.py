@torch.jit.script
def compute_reward(velocity: torch.Tensor, targets: torch.Tensor, torso_position: torch.Tensor, heading_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward shaping
    velocity_reward_temperature = 1.0
    heading_reward_temperature = 0.5

    # Compute the forward velocity reward (maximize forward velocity)
    forward_velocity_reward = velocity[:, 0]  # Assuming x-axis is the forward direction
    forward_velocity_reward_transformed = torch.exp(forward_velocity_reward * velocity_reward_temperature)

    # Compute the heading alignment reward (alignment with the target)
    heading_alignment_reward = heading_proj  # Measure of how well the ant is aligned with the target
    heading_alignment_reward_transformed = torch.exp(heading_alignment_reward * heading_reward_temperature)

    # Total reward as a weighted sum of components
    total_reward = forward_velocity_reward_transformed + heading_alignment_reward_transformed

    # Creating a dictionary of individual reward components for debugging and analysis
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "forward_velocity_reward_transformed": forward_velocity_reward_transformed,
        "heading_alignment_reward": heading_alignment_reward,
        "heading_alignment_reward_transformed": heading_alignment_reward_transformed,
    }

    return total_reward, reward_dict
