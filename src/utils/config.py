import yaml


def load_dqn_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config phai la mot dictionary.")

    # Kiểm tra các nhóm thông số bắt buộc
    for section in ["agent", "training", "logging"]:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Thieu hoac sai nhom config: {section}")

    agent = config["agent"]
    training = config["training"]

    if agent["learning_rate"] <= 0:
        raise ValueError("learning_rate phai lon hon 0.")

    if not 0 <= agent["gamma"] <= 1:
        raise ValueError("gamma phai nam trong khoang 0 den 1.")

    if not 0 <= agent["epsilon_end"] <= agent["epsilon_start"] <= 1:
        raise ValueError("Can 0 <= epsilon_end <= epsilon_start <= 1.")

    if not 0 < agent["batch_size"] <= agent["replay_buffer_size"]:
        raise ValueError("Batch size phai duong va khong vuot replay buffer.")

    if not 0 <= training["learning_starts"] < training["total_steps"]:
        raise ValueError("learning_starts phai nho hon total_steps va khong am.")

    return config