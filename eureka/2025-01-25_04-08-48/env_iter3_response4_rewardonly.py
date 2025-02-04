@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity difference to encourage fast running
    velocity_diff = potentials - prev_potentials

    # Adjust velocity reward's impact using a temperature parameter
    velocity_temp = 0.2  # Reduced temperature for reduced large-scale impact
    transformed_velocity_reward = torch.exp(velocity_diff * velocity_temp) - 1.0

    # Define and improve the uprightness reward component with adjusted temperature
    upright_temp = 2.0  # Increased temperature to enhance impact and sensitivity
    uprightness_penalty = (up_vec[:, 2] - 1.0).abs()
    consistency_reward = torch.exp(-uprightness_penalty * upright_temp) 

    # Combine the total reward with a new balance
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * consistency_reward  # Increased focus on stability

    # Collect each reward component for monitoring
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "consistency_reward": consistency_reward
    }

    return total_reward, reward_dict
