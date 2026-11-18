# Bàn giao tuần 1 — Người 3

## Phạm vi

Đã chuẩn bị:
- Cấu hình DQN ban đầu: configs/dqn_config.yaml.
- Hàm đọc và kiểm tra cơ bản config: src/utils/config.py.
- Logger TensorBoard: src/utils/logger.py.
- Chương trình kiểm tra logger bằng game thật: smoke_logger.py.

Chưa triển khai thuật toán huấn luyện trong phần bàn giao này.
Các giá trị DQN là đề xuất ban đầu, chưa được kiểm chứng qua huấn luyện.

## Cài đặt

Dùng Python 3.13 64-bit, chạy từ thư mục gốc repository:

```powershell
python -m pip install -r requirements-training.txt
```

## Kiểm tra config

```powershell
python -c "from src.utils.config import load_dqn_config; print(load_dqn_config('configs/dqn_config.yaml'))"
```

## Kiểm tra logger

```powershell
python smoke_logger.py
tensorboard --logdir logs --port 6006
```

Mở http://localhost:6006, chọn lượt chạy random_native_...

Chương trình chơi ngẫu nhiên 10 episode, mỗi episode tối đa
150 bước quyết định, sử dụng reward mặc định của môi trường.

Ba biểu đồ:
- Charts/Episode_Reward: tổng reward trong episode.
- Charts/Max_X_Position: tọa độ X xa nhất trong episode.
- Charts/Episode_Length: số bước quyết định trong episode.

Đây là dữ liệu chơi ngẫu nhiên, không phải kết quả huấn luyện.
File log được tạo tại logs/ và không đưa vào Git.

## Dùng trong chương trình huấn luyện

```python
from src.utils.config import load_dqn_config
from src.utils.logger import TrainingLogger

config = load_dqn_config("configs/dqn_config.yaml")
logger = TrainingLogger("logs/my_training_run")

try:
    # Gọi sau mỗi episode; các số dưới đây chỉ là ví dụ.
    logger.log_episode(
        episode=1,
        total_reward=100.0,
        max_x=250,
        steps=80,
    )
finally:
    logger.close()
```

## Lưu ý khi tích hợp

- Người 2 cần đọc và sử dụng config trong vòng lặp huấn luyện.
- smoke_logger.py có thông số chạy thử riêng, không sử dụng
  các thông số huấn luyện trong dqn_config.yaml.
- Lấy số hành động từ env.action_space.n.
- Các trường *_steps tính theo số lần env.step().
- target_update_interval_updates tính theo số lần cập nhật trọng số.
- Replay buffer khởi đầu có 10.000 trải nghiệm; cần đánh giá
  bộ nhớ trước khi tăng dung lượng.
- Logger hiện ghi ba chỉ số episode. Loss và epsilon sẽ được
  bổ sung khi kết nối với thuật toán huấn luyện.