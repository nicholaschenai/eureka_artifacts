@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_force_tensor: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor, dt: float):
    # Compute current speed towards the target
    velocity = root_states[:, 7:10]
    to_target = targets - root_states[:, 0:3]
    to_target[:, 2] = 0
    speed_towards_target = (velocity * torch.nn.functional.normalize(to_target, dim=-1)).sum(dim=-1)

    # Reward for moving quickly towards the target
    speed_reward_temp = 1.0
    speed_reward = torch.exp(speed_reward_temp * speed_towards_target) - 1.0

    # Reward for upright posture, using the projection of up vector in the vertical direction
    up_rightness_temp = 0.5
    up_reward = torch.exp(up_rightness_temp * (up_vec[:, 2] + 1.0) / 2.0) - 1.0

    # Penalize excessive forces in the joints
    force_penalty_temp = 0.1
    force_penalty = torch.exp(-force_penalty_temp * torch.norm(dof_force_tensor, p=2, dim=-1)) - 1.0

    # Total reward is the sum of the components
    total_reward = speed_reward + up_reward + force_penalty

    # Return the total reward and all individual components
    return total_reward, {
        "speed_reward": speed_reward,
        "up_reward": up_reward,
        "force_penalty": force_penalty
    }
