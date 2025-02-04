@torch.jit.script
def compute_reward(root_states: torch.Tensor, 
                   actions: torch.Tensor, 
                   dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant tensors
    velocity = root_states[:, 7:10]  # Linear velocity
    ang_velocity = root_states[:, 10:13]  # Angular velocity

    # Reward forward velocity (we assume x-axis corresponds to forward direction)
    forward_vel_reward = velocity[:, 0]  # Consider only the x component for forward movement

    # Penalize high angular velocity for stability
    stability_penalty = torch.norm(ang_velocity, p=2, dim=-1)

    # Penalize large actions to encourage energy efficiency
    action_penalty = torch.norm(actions, p=2, dim=-1)

    # Sum of rewards and penalties
    # Constants for weighing different components
    forward_vel_weight = 1.0
    stability_weight = 0.5
    action_weight = 0.1

    # Reward computation
    reward = (forward_vel_weight * forward_vel_reward 
              - stability_weight * stability_penalty 
              - action_weight * action_penalty)

    # Normalize and apply non-linearity for reward
    temperature_fw = 1.0
    temperature_stab = 1.0
    temperature_act = 1.0
    transformed_reward = torch.exp(reward / temperature_fw) - 1

    # Return the total reward and individual components for logging/analysis
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "stability_penalty": stability_penalty,
        "action_penalty": action_penalty
    }

    return transformed_reward, reward_dict
