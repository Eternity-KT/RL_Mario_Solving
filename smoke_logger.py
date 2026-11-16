from datetime import datetime

from src.env import make_mario_env
from src.utils.logger import TrainingLogger


run_name = datetime.now().strftime("random_native_%Y%m%d_%H%M%S")
log_dir = f"logs/{run_name}"

logger = TrainingLogger(log_dir)

try:
    with make_mario_env(max_episode_steps=150) as env:
        # Đặt seed riêng cho bộ chọn hành động ngẫu nhiên
        env.action_space.seed(42)

        for episode in range(1, 11):
            obs, info = env.reset(seed=42)

            total_reward = 0.0
            max_x = info["x_pos"]
            steps = 0

            while True:
                # Mỗi bước chọn ngẫu nhiên một hành động hợp lệ
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                max_x = max(max_x, info["x_pos"])
                steps += 1

                if terminated or truncated:
                    break

            # Chỉ ghi thống kê sau khi lượt chơi kết thúc
            logger.log_episode(
                episode=episode,
                total_reward=total_reward,
                max_x=max_x,
                steps=steps,
            )

            print(
                f"Episode {episode:2d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Max X: {max_x:4d} | "
                f"Steps: {steps}"
            )

finally:
    logger.close()

print("\nLog da luu tai:", log_dir)