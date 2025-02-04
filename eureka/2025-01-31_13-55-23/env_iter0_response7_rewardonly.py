@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor, contact_forces: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant states
    velocity = root_states[:, 7:10]
    torso_position = root_states[:, 0:3]

    # Define constants and temperature parameters for transformations
    velocity_weight = 1.0
    energy_weight = 0.005
    stability_weight = 0.1

    velocity_temp = 10.0
    energy_temp = 0.1
    stability_temp = 10.0

    # Compute the forward velocity (assuming forward is along x-axis)
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity
    forward_velocity_reward = forward_velocity * velocity_weight

    # Penalty for large actions (energy usage)
    energy_penalty = torch.sum(actions ** 2, dim=1) * energy_weight

    # Penalty for high contact forces (instability)
    stability_penalty = torch.norm(contact_forces, p=2, dim=-1) * stability_weight

    # Transform rewards and penalties using exponential function and a temperature parameter
    forward_velocity_reward = torch.exp(forward_velocity_reward / velocity_temp)
    energy_penalty = torch.exp(-energy_penalty / energy_temp)
    stability_penalty = torch.exp(-stability_penalty / stability_temp)

    # Total reward composition
    total_reward = forward_velocity_reward - energy_penalty - stability_penalty

    # Creating a reward dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty": -energy_penalty,
        "stability_penalty": -stability_penalty
    }

    return total_reward, reward_dict
