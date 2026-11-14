# SỔ TAY KỸ THUẬT & QUY TRÌNH LÀM VIỆC CHI TIẾT - NGƯỜI 1
## VAI TRÒ: CHUYÊN GIA MÔI TRƯỜNG, TIỀN XỬ LÝ DỮ LIỆU & NỀN TẢNG TRIỂN KHAI (ENVIRONMENT & PREPROCESSING LEAD)

---

## I. THÔNG TIN CHUNG & VAI TRÒ TRÁCH NHIỆM

- **Họ và tên thành viên:** [Điền tên Thành viên 1]
- **Vai trò chuyên trách:** Kỹ sư Môi trường & Tiền xử lý dữ liệu (Environment & Preprocessing Engineer).
- **Trách nhiệm cốt lõi:**
  1. Xây dựng môi trường giả lập NES Super Mario Bros tương thích 100% với chuẩn **Gymnasium** hiện đại nhất.
  2. Thiết kế và triển khai toàn bộ pipeline tiền xử lý khung hình (Grayscale, Resize $84 \times 84$, FrameSkip = 4, FrameStack = 4).
  3. Chuẩn hóa không gian hành động rời rạc (JoypadSpace sang 7 hành động tối ưu `SIMPLE_MOVEMENT`).
  4. Đóng gói hàm khởi tạo môi trường tiêu chuẩn `make_mario_env()` để toàn bộ nhóm (Người 2, 3, 4) sử dụng thống nhất.
  5. Xây dựng bộ script baseline kiểm thử môi trường (FPS, mức tiêu hao RAM/VRAM, tính ổn định qua hàng triệu steps).
  6. Soạn thảo các chương liên quan đến Môi trường và Tiền xử lý trong Báo cáo học thuật (Chương 1, 3, 9) và thuyết trình phần tương ứng trên Slide.

---

## II. BẢN ĐỒ KIẾN THỨC & CÔNG CỤ CẦN THÀNH THẠO

### 1. Kiến thức lý thuyết bắt buộc
- **Cơ chế hoạt động của Máy chơi game NES (Nintendo Entertainment System):**
  - Tần số khung hình gốc: 60 FPS (NTSC).
  - Độ phân giải gốc: $256 \times 240$ pixels, 3 kênh màu RGB.
  - Hiện tượng chớp hình (Sprite Flickering): Chip PPU của NES chỉ vẽ được tối đa 8 sprites trên một đường quét ngang, dẫn đến các vật thể nhấp nháy luân phiên giữa 2 khung hình liên tiếp. Cần cơ chế lấy giá trị cực đại (Max-pooling) giữa 2 frame cuối cùng khi skip frame.
- **Chuẩn giao tiếp Gymnasium vs Gym truyền thống:**
  - `reset()`: Gym cũ trả về `obs`, Gymnasium trả về `(obs, info)`.
  - `step(action)`: Gym cũ trả về `obs, reward, done, info` (4 giá trị). Gymnasium trả về `obs, reward, terminated, truncated, info` (5 giá trị). Trong đó `terminated` là kết thúc do luật chơi (chết hoặc thắng), `truncated` là kết thúc do giới hạn thời gian (time limit).
- **Lý thuyết Markov Decision Process (MDP) trong xử lý hình ảnh:**
  - Khung hình đơn lẻ là POMDP (Partially Observable MDP) vì không thể hiện vận tốc.
  - Frame Stacking (ghép 4 khung hình) khôi phục tính chất Markov đầy đủ: $P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t, s_{t-1}, ..., s_0, a_t)$.

### 2. Công cụ và Thư viện sử dụng
- `gym-super-mario-bros` (v7.4.0) & `nes-py` (v8.2.1)
- `gymnasium` (v0.29.1 hoặc v1.0.0+) & `shimmy`
- `opencv-python` (`cv2`) cho việc resize và nội suy bilinear ảnh nhanh
- `numpy` cho các phép biến đổi mảng đa chiều
- `torchvision.transforms` (nếu tiền xử lý trực tiếp trên GPU)

---

## III. THIẾT KẾ KỸ THUẬT & CODE TEMPLATES CỦA BỘ TIỀN XỬ LÝ (WRAPPERS)

Người 1 chịu trách nhiệm viết file `src/env/wrappers.py` với cấu trúc chuẩn mực sau:

### 1. Custom Wrapper tương thích Gymnasium: `GymnasiumCompatibilityWrapper`
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """
    Chuyển đổi môi trường nes-py/gym cũ sang chuẩn Gymnasium (5 giá trị trả về trong step).
    """
    def __init__(self, env):
        super().__init__(env)
        # Giữ nguyên observation_space và action_space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=env.observation_space.shape, dtype=np.uint8
        )
        self.action_space = spaces.Discrete(env.action_space.n)

    def reset(self, **kwargs):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            return obs
        info = {}
        return obs, info

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Tách cờ done thành terminated (chết hoặc thắng cờ) và truncated (hết giờ)
        terminated = done
        truncated = False
        if info.get('time', 1) <= 0:
            truncated = True
            terminated = False
        return state, float(reward), terminated, truncated, info
```

### 2. Wrapper giảm chiều dữ liệu: `GrayScaleAndResizeWrapper`
```python
import cv2

class GrayScaleAndResizeWrapper(gym.ObservationWrapper):
    """
    Chuyển đổi ảnh màu 240x256x3 sang ảnh xám 84x84x1 để tối ưu bộ nhớ và tính toán.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # Chuyển đổi RGB sang Grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize về 84x84 sử dụng phép nội suy diện tích (INTER_AREA)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)
```

### 3. Wrapper FrameSkip & FrameStack: `CustomFrameSkipAndStackWrapper`
```python
from collections import deque

class FrameSkipAndStackWrapper(gym.Wrapper):
    """
    Bỏ qua k khung hình (FrameSkip=4), lấy max giữa 2 frame cuối để loại bỏ flickering,
    và xếp chồng k khung hình liên tiếp tạo thành tensor kích thước (4, 84, 84).
    """
    def __init__(self, env, skip=4, stack=4):
        super().__init__(env)
        self.skip = skip
        self.stack = stack
        self.obs_buffer = deque(maxlen=2)
        self.frames = deque(maxlen=stack)
        
        # Không gian quan sát đầu ra có shape là (stack, 84, 84) - Channel First cho PyTorch
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(stack, 84, 84), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # obs có shape (84, 84, 1) -> squeeze thành (84, 84)
        obs_sq = np.squeeze(obs, axis=-1)
        for _ in range(self.stack):
            self.frames.append(obs_sq)
        return np.array(self.frames, dtype=np.uint8), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            self.obs_buffer.append(np.squeeze(obs, axis=-1))
            if term or trunc:
                terminated = term
                truncated = trunc
                break

        # Max pooling giữa 2 khung hình cuối để xử lý lỗi nhấp nháy hình NES
        if len(self.obs_buffer) == 2:
            max_frame = np.maximum(self.obs_buffer[0], self.obs_buffer[1])
        else:
            max_frame = self.obs_buffer[0]

        self.frames.append(max_frame)
        stacked_obs = np.array(self.frames, dtype=np.uint8)
        return stacked_obs, total_reward, terminated, truncated, info
```

### 4. Hàm đóng gói tổng thể: `src/env/make_env.py`
```python
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from src.env.wrappers import (
    GymnasiumCompatibilityWrapper,
    GrayScaleAndResizeWrapper,
    FrameSkipAndStackWrapper
)

def make_mario_env(world=1, stage=1, action_type='SIMPLE'):
    """
    Hàm chuẩn hóa để toàn bộ nhóm khởi tạo môi trường Mario đồng nhất.
    """
    env_id = f'SuperMarioBros-{world}-{stage}-v0'
    env = gym_super_mario_bros.make(env_id)
    
    # 1. Rút gọn không gian hành động
    if action_type == 'RIGHT_ONLY':
        env = JoypadSpace(env, RIGHT_ONLY)
    else:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
    # 2. Bọc tương thích Gymnasium
    env = GymnasiumCompatibilityWrapper(env)
    
    # 3. Tiền xử lý ảnh xám và Resize 84x84
    env = GrayScaleAndResizeWrapper(env, width=84, height=84)
    
    # 4. Skip frame 4 và Stack frame 4
    env = FrameSkipAndStackWrapper(env, skip=4, stack=4)
    
    return env
```

---

## IV. LỘ TRÌNH THỰC HIỆN CHI TIẾT THEO TỪNG TUẦN (WEEKLY SPRINT)

| Tuần | Mục tiêu tuần | Danh sách đầu việc cụ thể (Action Items) | File tạo ra / Chỉnh sửa | Bàn giao cho ai | Tiêu chí hoàn thành (DoD) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Tuần 1** | Thiết lập môi trường & Wrappers | - Nghiên cứu nes-py, gymnasium, giải quyết lỗi build MSVC trên Windows.<br>- Viết `requirements.txt` cố định version thư viện.<br>- Hoàn thiện `wrappers.py` và `make_env.py`.<br>- Viết script `baseline_random.py` test FPS. | `src/env/wrappers.py`<br>`src/env/make_env.py`<br>`requirements.txt`<br>`baseline_random.py` | Cả nhóm (Người 2, 3, 4) | - Môi trường chạy $\ge 60$ FPS.<br>- Tensor output chuẩn `(4, 84, 84)`.<br>- 4 máy cài đặt thành công. |
| **Tuần 2** | Hỗ trợ tích hợp Alpha & Viết báo cáo | - Phối hợp Người 2 kiểm tra tensor flow vào mạng PyTorch.<br>- Hỗ trợ Người 3 tích hợp `CustomRewardWrapper`.<br>- Viết bản thảo Chương 1 (Tổng quan) và Chương 3 (Mô tả bài toán & Tiền xử lý). | `docs/chapter1.md`<br>`docs/chapter3.md`<br>`tests/test_env.py` | Người 4 (Tổng hợp báo cáo) | - Không phát sinh lỗi kích thước tensor.<br>- Hoàn thành 100% bản thảo 2 chương. |
| **Tuần 3** | Tối ưu hóa FPS & Sửa lỗi bẫy môi trường | - Kiểm tra hiện tượng ngốn RAM khi chạy môi trường qua $1.0\times 10^6$ steps.<br>- Phối hợp với Người 3 phân tích dữ liệu tọa độ X, thời gian game để nhận diện bẫy kẹt ống nước.<br>- Rà soát điều kiện cờ chiến thắng (`flag_get == True`). | `src/env/wrappers.py`<br>`src/env/make_env.py` | Người 3 | - Mức tiêu hao RAM ổn định $< 1.5$ GB.<br>- Cơ chế nhận diện thắng cờ hoạt động chuẩn xác 100%. |
| **Tuần 4** | Triển khai môi trường cho thí nghiệm mở rộng | - Tạo môi trường với biến thể hành động `RIGHT_ONLY` phục vụ Ablation Study của Người 3.<br>- Thử nghiệm tạo môi trường World 1-2 để đánh giá khả năng tổng quát hóa (Generalization).<br>- Viết bản thảo Chương 9 (Hạn chế & Hướng phát triển). | `src/env/make_env.py`<br>`docs/chapter9.md` | Người 2, 3, 4 | - Cung cấp đủ 2 biến thể môi trường cho thí nghiệm 2 điểm cộng.<br>- Hoàn thành bản thảo Chương 9. |
| **Tuần 5** | Rà soát kỹ thuật, Hoàn thiện Báo cáo & Luyện tập Slide | - Rà soát toàn bộ docstrings của module `src/env/`.<br>- Chuẩn hóa các hình vẽ sơ đồ tiền xử lý khung hình cho Báo cáo và Slide.<br>- Luyện tập bài nói 5 phút về Môi trường và Tiền xử lý.<br>- Tham gia 2 buổi Mock Defense. | Mã nguồn sạch sẽ,<br>Slide phần 1,<br>Nội dung thuyết trình | Cả nhóm | - Trả lời lưu loát 100% các câu hỏi về môi trường và MDP.<br>- Thuyết trình đúng 5 phút. |

---

## V. NHỮNG LỖI THƯỜNG GẶP VÀ CÁCH KHẮC PHỤC TRIỆT ĐỂ (TROUBLESHOOTING)

1. **Lỗi `Microsoft Visual C++ 14.0 or greater is required` khi cài `nes-py` trên Windows:**
   - *Nguyên nhân:* `nes-py` chứa mã nguồn C++ của giả lập FCEU. Khi cài đặt trên Python phiên bản mới (3.12, 3.13) mà chưa có pre-built binary wheel, pip sẽ tự biên dịch và yêu cầu bộ build tools C++.
   - *Giải pháp:*
     - *Cách 1 (Khuyến nghị số 1):* Tạo môi trường ảo với Python 3.10 hoặc 3.11 (`conda create -n mario_rl python=3.10 -y`), ở phiên bản này `nes-py` đã có wheel dựng sẵn, cài đặt chỉ trong 5 giây.
     - *Cách 2:* Cài đặt gói `Visual Studio Build Tools` chọn workload `Desktop development with C++`.
2. **Lỗi `ValueError: not enough values to unpack (expected 5, got 4)` khi gọi `env.step()`:**
   - *Nguyên nhân:* Do thư viện `gym-super-mario-bros` vốn được viết theo chuẩn OpenAI Gym cũ (`step()` chỉ trả về 4 biến: `obs, reward, done, info`), trong khi code vòng lặp hoặc Stable-Baselines3 / Gymnasium mới yêu cầu 5 biến (`obs, reward, terminated, truncated, info`).
   - *Giải pháp:* Người 1 bắt buộc phải bọc môi trường qua `GymnasiumCompatibilityWrapper` như đã lập trình ở mục III.1.
3. **Hiện tượng rò rỉ bộ nhớ (Memory Leak) sau vài trăm episode:**
   - *Nguyên nhân:* Trong các wrapper có lưu trữ `deque` hoặc danh sách frames nhưng không giải phóng đúng cách khi gọi `reset()`, hoặc lưu toàn bộ ảnh float32 thay vì uint8.
   - *Giải pháp:* Luôn ép kiểu ảnh xám về `np.uint8` (1 byte mỗi pixel thay vì 4 byte float). Gọi `self.obs_buffer.clear()` và khởi tạo lại deque trong hàm `reset()`.

---

## VI. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 1

Khi hội đồng phản biện chất vấn về phần Môi trường và Tiền xử lý, Người 1 cần trả lời tự tin và chính xác theo các gợi ý sau:

- **Câu hỏi 1: Tại sao không dùng trực tiếp ảnh màu RGB $240 \times 256$ mà phải chuyển về Grayscale $84 \times 84$?**
  - *Trả lời:* Ảnh màu $240 \times 256 \times 3$ có tới $184,320$ giá trị đầu vào cho mỗi khung hình. Khi chuyển về ảnh xám $84 \times 84 \times 1$, số điểm ảnh chỉ còn $7,056$ giá trị (giảm hơn $96\%$ kích thước dữ liệu). Trong Super Mario Bros, các thông tin sống còn để điều khiển tác nhân (vị trí Mario, khoảng cách tới quái vật Goomba, mép hố sâu, ống nước) đều được biểu diễn đầy đủ qua sự tương phản hình khối và độ sáng, màu sắc của gạch hay áo Mario không đem lại thêm giá trị quyết định hành vi. Việc giảm chiều giúp giảm hàng chục triệu tham số của mạng CNN, tăng tốc độ tính toán gấp $5-10$ lần và hạn chế nguy cơ quá khớp (overfitting).
- **Câu hỏi 2: Giải thích cơ chế Frame Skip = 4. Tại sao bỏ qua 4 khung hình mà Mario không bị đâm vào quái vật?**
  - *Trả lời:* NES chạy ở tần số 60 FPS, nghĩa là cứ khoảng 16.6 mili-giây lại có một khung hình mới. Với tốc độ di chuyển của Mario và Goomba, sự biến đổi vị trí trong 1-2 khung hình là cực kỳ nhỏ (vài pixel). Tác nhân không cần thiết và cũng không thể đưa ra một chiến lược mới sau mỗi 16ms. Bằng cách lặp lại hành động đó trong 4 khung hình liên tiếp và chỉ tính toán mạng nơ-ron ở khung hình thứ 4, tốc độ huấn luyện thực tế tăng lên gần 4 lần mà tác nhân vẫn đảm bảo phản xạ tốt trong khoảng thời gian $\approx 67$ms (nhanh hơn rất nhiều so với thời gian phản xạ trung bình của con người là $200-250$ms).
- **Câu hỏi 3: Không gian hành động `SIMPLE_MOVEMENT` gồm những hành động gì và tại sao không dùng toàn bộ nút của tay cầm NES?**
  - *Trả lời:* Tay cầm NES gốc có 8 nút bấm (A, B, Select, Start, Up, Down, Left, Right), tạo ra $2^8 = 256$ tổ hợp hành động khả dĩ. Tuy nhiên phần lớn các tổ hợp là vô nghĩa hoặc không hợp lệ (ví dụ: bấm đồng thời Left và Right, hoặc bấm Select/Start). `SIMPLE_MOVEMENT` cô đọng thành 7 hành động tối cần thiết để chơi game: Đứng yên, Sang phải, Nhảy phải, Chạy nhanh phải, Chạy nhanh nhảy phải, Nhảy tại chỗ, và Sang trái. Việc giảm không gian hành động từ 256 xuống 7 giúp không gian tìm kiếm chính sách thu hẹp cấp số nhân, giúp thuật toán Q-Learning hội tụ nhanh chóng trong thời gian 5 tuần.

---

## VII. CHECKLIST TỰ ĐÁNH GIÁ TIẾN ĐỘ CỦA NGƯỜI 1

- [ ] Môi trường ảo hoạt động trơn tru trên cả Windows và Linux.
- [ ] Không xuất hiện lỗi xung đột `Gymnasium` vs `Gym` trong toàn bộ pipeline.
- [ ] Hàm `make_mario_env()` trả về đúng tensor shape `(4, 84, 84)` kiểu `np.uint8`.
- [ ] Môi trường chạy baseline đạt tốc độ tối thiểu 60 FPS trên CPU thông thường.
- [ ] RAM tiêu thụ ổn định dưới 1.5 GB sau 100,000 steps liên tục.
- [ ] Cung cấp đầy đủ tài liệu hướng dẫn và hỗ trợ 3 thành viên còn lại cài đặt thành công.
- [ ] Hoàn thành bản thảo Chương 1, Chương 3 và Chương 9 của Báo cáo đồ án.
- [ ] Hoàn thành phần Slide và thuyết trình thử nghiệm tự tin trong 5 phút.
