@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Ensure device consistency
    device = franka_grasp_pos.device

    # Reward component for minimizing distance between the robot's grasp position and the drawer's grasp position
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward_temperature = 0.1
    distance_reward = torch.exp(-distance / distance_reward_temperature)

    # Reward component for increasing the cabinet door position (open it further)
    door_opening_reward = cabinet_dof_pos[:, 3]  # Assuming the 4th DOF corresponds to the cabinet door position
    door_opening_reward_temperature = 0.05
    door_opening_reward_transformed = torch.exp(door_opening_reward / door_opening_reward_temperature)

    # Reward component for increasing the cabinet door velocity (encouraging opening motion)
    door_velocity_reward = torch.abs(cabinet_dof_vel[:, 3])
    door_velocity_reward_temperature = 0.1
    door_velocity_reward_transformed = torch.exp(door_velocity_reward / door_velocity_reward_temperature)

    # Total reward is a weighted sum of the components
    total_reward = distance_reward + door_opening_reward_transformed + door_velocity_reward_transformed

    # Compile individual reward components into a dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "door_opening_reward": door_opening_reward_transformed,
        "door_velocity_reward": door_velocity_reward_transformed
    }

    return total_reward, reward_components
