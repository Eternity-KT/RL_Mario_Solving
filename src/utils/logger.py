from tensorboardX import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_episode(self, episode, total_reward, max_x, steps):
        self.writer.add_scalar(
            "Charts/Episode_Reward", total_reward, episode
        )

        self.writer.add_scalar(
            "Charts/Max_X_Position", max_x, episode
        )

        self.writer.add_scalar(
            "Charts/Episode_Length", steps, episode
        )

        # Đẩy dữ liệu đang chờ xuống file
        self.writer.flush()

    def close(self):
        self.writer.close()