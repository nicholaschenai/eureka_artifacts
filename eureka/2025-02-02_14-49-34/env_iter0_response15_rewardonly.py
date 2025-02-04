@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance to the target
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, p=2, dim=-1)
    distance_reward = -distance_to_drawer

    # Calculate the reward based on the cabinet door position
    # Assumes the target cabinet_dof_pos[3] should be maximized (fully opened state)
    door_opened_reward = cabinet_dof_pos[:, 3]

    # Introduce a velocity reward to encourage the door to continue moving once motion is detected
    velocity_reward = cabinet_dof_vel[:, 3]

    # Parameters for transforming the rewards
    distance_temperature = 1.0
    door_temperature = 1.0
    velocity_temperature = 1.0

    # Transforming each reward component using a soft exponential
    distance_reward_transformed = torch.exp(distance_reward / distance_temperature)
    door_opened_reward_transformed = torch.exp(door_opened_reward / door_temperature)
    velocity_reward_transformed = torch.exp(velocity_reward / velocity_temperature)

    # Combine the reward components
    total_reward = distance_reward_transformed + door_opened_reward_transformed + velocity_reward_transformed

    # Create a dictionary containing each individual reward component for debugging/analysis purposes
    reward_dict = {
        "distance_reward": distance_reward_transformed,
        "door_opened_reward": door_opened_reward_transformed,
        "velocity_reward": velocity_reward_transformed
    }

    return total_reward, reward_dict
