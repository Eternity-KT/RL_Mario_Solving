# Bàn giao phần môi trường — Người 1

## Phiên bản đã chọn

CPython **3.13, 64-bit**. `requirements.txt` ghim Mario 9.1.0,
nes-py 9.0.1, Gymnasium 1.2.3, NumPy 2.3.3 và Pillow 11.3.0.
Đây là bộ thư viện Gymnasium-native, thay cho bộ Gym cũ trong tài liệu kế hoạch.
Không cài thêm `gym`, `shimmy` hoặc dùng code compatibility wrapper cũ.
`psutil` phục vụ đo RAM trong baseline; `pytest` nằm trong requirements-dev.txt.
PyTorch, PPO, TensorBoard và công cụ xuất báo cáo Word thuộc các phần triển khai
tiếp theo, chưa phải phụ thuộc của môi trường.

Nguồn tham khảo: [Mario](https://pypi.org/project/gym-super-mario-bros/9.1.0/),
[nes-py](https://pypi.org/project/nes-py/9.0.1/),
[quy tắc kết thúc episode của Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Pipeline và đơn vị

```text
Mario World 1-1 RGB (240, 256, 3)
  -> JoypadSpace (7, 5 hoặc 2 hành động)
  -> Lặp hành động tối đa 4 frame; cộng reward; max 2 frame RGB cuối
  -> TimeLimit: tối đa 4500 bước quyết định
  -> Reward wrapper tùy chọn do Người 3 cung cấp
  -> Pillow grayscale + resize BOX (84, 84)
  -> FrameStackObservation: 4 ảnh đã xử lý
  -> RecordEpisodeStatistics
```

- Một `env.step(action)` là **một bước quyết định của agent**. Mặc định giả lập
  chạy tối đa 4 frame, dừng ngay nếu thắng, chết hoặc bị ngắt.
- Stack chứa bốn ảnh sau bốn lượt ra quyết định, theo thứ tự cũ đến mới.
  Lúc reset, lặp ảnh đầu bốn lần. Stack giúp suy ra chuyển động; không bảo đảm
  quan sát đầy đủ mọi trạng thái ẩn của trò chơi.
- Max-pooling diễn ra trước grayscale/resize. Sao chép frame từ bộ đệm giả lập
  để ảnh cũ không bị thay đổi khi giả lập chạy tiếp. Không pool qua episode
  hoặc qua lượt quyết định nếu lượt hiện tại chỉ thu được một frame.
- Đây là một môi trường đơn. Người 2 có thể tạo nhiều instance độc lập nếu
  cần vector hóa; không giả định có sẵn batch dimension ở observation.

## Hợp đồng bàn giao

```python
from src.env import make_mario_env

env = make_mario_env(action_type="SIMPLE_MOVEMENT")
try:
    obs, info = env.reset(seed=42)
    env.action_space.seed(42)  # riêng cho random agent
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
finally:
    env.close()
```

| Thành phần | Quy ước |
| --- | --- |
| Observation | NumPy `uint8`, `(4, 84, 84)`, khoảng 0–255, không batch dimension |
| Action | `int` thuộc `env.action_space`; SIMPLE_MOVEMENT có 7, RIGHT_ONLY có 5, RIGHT_JUMP có 2 |
| Reward mặc định | Tổng reward native của các frame thực sự đã chạy; chưa thêm shaping riêng |
| `terminated` | Giữ nguyên luật kết thúc native của màn chơi, gồm thắng/chết |
| `truncated` | Giới hạn ngoài của wrapper; không sửa cờ chỉ dựa trên đồng hồ trong game |
| Reset | Truyền `seed`, `options` xuống môi trường; bỏ toàn bộ lịch sử stack |
| Render | Chọn `render_mode` khi tạo; `env.render()` không nhận thêm `mode` |
| RGB render | `render_mode="rgb_array"`: ảnh gốc `(240, 256, 3)`, tách khỏi observation |
| Human render | `render_mode="human"`: cửa sổ native; không bật khi benchmark headless |

Mapping SIMPLE_MOVEMENT: 0 NOOP, 1 right, 2 right+A, 3 right+B,
4 right+A+B, 5 A, 6 left. RIGHT_ONLY dùng năm mục đầu.

Mapping `RIGHT_JUMP`: **0 right, 1 right+A**. Đây là tập hành động tùy chỉnh
theo [ví dụ Mario của PyTorch](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#initialize-environment),
khác với `RIGHT_ONLY` có 5 hành động. Cả hai action đều giữ nút phải; không
có NOOP, đi trái hoặc chạy nhanh bằng B. Các quy ước frame skip, reward,
seed và đầu ra ảnh giữ như pipeline chung.

```python
env = make_mario_env(action_type="RIGHT_JUMP")
try:
    obs, info = env.reset(seed=42)
    action_count = env.action_space.n  # 2; dùng để cấu hình đầu ra của mạng
finally:
    env.close()
```

Có thể đọc `configs/env_two_actions.json` và truyền `**config` vào factory.
`configs/env.json` vẫn là cấu hình 7 hành động. Khi đổi tập hành động, cần
tạo lại đầu ra mạng tương ứng và huấn luyện mô hình cho tập đó; không dùng
trực tiếp checkpoint 5/7 hành động để đánh giá bản 2 hành động.

`info` giữ các trường native như `x_pos`, `flag_get`, `time`, `life`, `status`
và bổ sung:

- `frames_this_step`: số frame thực chạy trong lượt này (0 lúc reset).
- `episode_emulator_frames`: tổng số frame của episode, tính từ reset.
- `default_reward`: tổng reward native của lượt, trước reward wrapper riêng.
- `reward_components_sum`: tổng các thành phần reward native chưa clipping
  theo từng frame. Tổng này có thể khác `default_reward` vì native có clipping.
- `episode`: chỉ xuất hiện khi kết thúc; `r` là tổng reward sau shaping,
  `l` là số bước quyết định, `t` là số giây chạy episode.

Không chuẩn hóa ảnh sang float trong môi trường hoặc replay buffer. Người 2
chuyển batch sang float và chia 255 tại đầu vào mạng. Giữ riêng `terminated`
và `truncated` khi lưu transition: reset khi một trong hai đúng; bootstrap
Double DQN bị chặn bởi `terminated`, không chặn chỉ vì `truncated`.
Nếu vector hóa tự reset, phải dùng observation cuối episode cho bootstrap.

## Gắn reward của Người 3

```python
env = make_mario_env(reward_wrapper=lambda base: MyRewardWrapper(base))
```

Wrapper reward chạy **một lần mỗi bước quyết định**, nhận tổng reward native
và info cuối lượt; thấy được cờ TimeLimit. Các hệ số phạt thời gian, ngưỡng
chống kẹt phải dùng cùng đơn vị này. Nếu muốn tính theo frame, dùng
`info['frames_this_step']`. `reset()` của reward wrapper cũng phải xóa lịch sử
tọa độ/counter và truyền tiếp seed/options. Không dùng `terminated` để biểu
diễn việc cưỡng chế dừng do anti-stuck bên ngoài.

Factory chưa gắn shaping mặc định: baseline dùng chính reward của phiên bản
Mario đã ghim. Default reward vốn có nhiều thành phần, không gọi là sparse
reward. Khi so sánh reward, đánh giá theo `flag_get` và tiến độ X thay vì so
trực tiếp tổng reward có thang đo khác nhau.

## Đánh giá và tái lập

Seed reset được truyền tới giả lập, nhưng màn 1-1 khôi phục cùng trạng thái
ban đầu; thay seed không bảo đảm tạo ra lượt chơi khác. Kiểm thử cùng seed +
cùng chuỗi action phải lặp lại được, không có nghĩa các seed khác nhau là
các mẫu kiểm thử độc lập. Cần nhiều seed huấn luyện và một giao thức tạo biến
thiên đánh giá được thống nhất riêng nếu muốn kết luận về độ ổn định.

## Kiểm thử và benchmark

```powershell
python -m pytest
python -m pytest -m "not integration"
python baseline_random.py --steps 10000
python baseline_random.py --steps 100000 --output results/baseline_100k.json
python baseline_random.py --action-type RIGHT_ONLY --steps 10000
python baseline_random.py --action-type RIGHT_JUMP --steps 1000
python baseline_random.py --config configs/env_two_actions.json --steps 1000
python baseline_random.py --steps 1000 --capture recordings/mario_frame.png
python baseline_random.py --render --steps 1000
```

Script mặc định chạy headless, warmup 100 quyết định trước khi đo; ghi rõ
`agent_steps_per_second` và `emulator_frames_per_second`. Thời gian đo có cả
reset giữa các episode, không bao gồm khởi tạo/warmup. Báo cáo JSON chứa
cấu hình, seed, phiên bản, nền tảng, max X, số episode và mẫu RSS mỗi 256 bước.
RSS là bộ nhớ tiến trình Python gồm thư viện/môi trường, không gồm replay
buffer hay mô hình. Peak là giá trị lớn nhất **đã lấy mẫu**. Một lượt ngắn
không chứng minh không có memory leak; không đặt ngưỡng FPS chung cho mọi máy.

Test giả lập nhỏ kiểm tra ảnh dùng chung bộ nhớ, reset, reward sum, kết thúc
sớm và phân biệt hai cờ. Integration test dùng game thật kiểm tra Gymnasium
contract, action spaces, stack, giới hạn bước, seed, reward hook và RGB render.
Gymnasium checker kiểm tra reset/step/seed/space của pipeline; bỏ phép thử
clone để kiểm tra close vì `JoypadSpace` của thư viện không ghi constructor
arguments. Dùng `make_mario_env(...)` để tạo lại môi trường, không dùng
`env.spec.make()` để clone pipeline này. Mỗi nơi gọi vẫn phải `close()` hoặc
dùng context manager như ví dụ.

Nếu `pip` cần build nes-py trên Windows, cần Visual Studio Build Tools với
Desktop development with C++. Không mặc định rằng đổi Python sẽ luôn có wheel.
Thử `python -m pip install --only-binary=:all: -r requirements.txt` nếu chỉ
muốn cài wheel; lỗi thiếu wheel phải xử lý theo nền tảng/Python thực tế.

## Phần việc tiếp theo

- Người 2: `src/models/`, `src/agents/`, replay buffer và vòng lặp train.
- Người 3: `src/rewards/`, cấu hình train và thí nghiệm reward.
- Người 4: evaluation độc lập, video MP4, báo cáo và release.
- `checkpoints/`, `recordings/`, `results/` là thư mục đầu ra; log train có thể
  đặt ở `logs/`. Các kết quả sinh tự động không mặc định đưa vào Git.
