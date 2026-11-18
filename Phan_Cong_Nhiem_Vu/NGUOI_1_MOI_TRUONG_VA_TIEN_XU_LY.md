# SỔ TAY KỸ THUẬT & PHÂN CÔNG TÁC CHIẾN 14 NGÀY - NGƯỜI 1
## VAI TRÒ: CHUYÊN GIA MÔI TRƯỜNG & TIỀN XỬ LÝ DỮ LIỆU (ENVIRONMENT & PREPROCESSING LEAD)
### Đồ án: Dạy Super Mario Bros vượt màn bằng PPO (PyTorch) - Học phần Học tăng cường

---

## I. THÔNG TIN CHUNG & TRÁCH NHIỆM CỐT LÕI

- **Họ và tên thành viên:** [Điền tên Thành viên 1]
- **Vai trò chuyên trách:** Chuyên gia Môi trường & Tiền xử lý dữ liệu.
- **Tệp mã nguồn sở hữu chính:** `mario_ppo/wrappers.py` (Toàn quyền tác giả).
- **Tệp mã nguồn phối hợp:** `mario_ppo/env.py` (Phối hợp với Người 4 để thiết lập chuỗi wrapper và mẹo huấn luyện).
- **Bộ bài kiểm tra đơn vị cá nhân:** `tests/test_wrappers.py` (Bắt buộc PASS 7/7 bài test).
- **Mục tiêu điểm số đóng góp:** Đạt trọn vẹn 6/6 điểm phần Wrappers trong Hạng mục A (Cài đặt thuật toán); đóng góp Mục 1 và Mục 5 trong Báo cáo học thuật (25 điểm).

---

## II. ĐẶC TẢ KỸ THUẬT CÁC HÀM CẦN CÀI ĐẶT TRONG `mario_ppo/wrappers.py`

Người 1 chịu trách nhiệm hiện thực hóa 3 lớp Wrapper kế thừa từ Gymnasium trong `mario_ppo/wrappers.py`. Đây là tầng tiếp xúc trực tiếp với môi trường `mo-supermario-v0`.

```
[Môi trường NES mo-supermario-v0] (Frame RGB 240x256x3, Reward vector 5 chiều)
       │
       ▼
[DeathPenaltyFix] (Có sẵn: Vá lỗi phạt chết -25)
       │
       ▼
[XPosWrapFix] (Có sẵn: Vá lỗi tràn tọa độ x 16-bit)
       │
       ▼
[ButtonActions] (Có sẵn: Ánh xạ bitmask NES sang 7 phím tập 'simple')
       │
       ▼
[FrameSkip(skip=4)] ───> [NGƯỜI 1 CÀI ĐẶT: Max 2 frame cuối, cộng dồn vector reward]
       │
       ├── [NẾU HUẤN LUYỆN (if not eval_mode)]:
       │     ├─ [Mẹo: NoProgressTruncation] ──> [NGƯỜI 1 VIẾT BỔ SUNG Ở CUỐI TỆP]
       │     └─ [RandomNoopStart(0..30)] (Có sẵn)
       │
       ▼
[ScalarizeReward(weights)] ───> [NGƯỜI 1 CÀI ĐẶT: Tích vô hướng w · r đổi vector thành số thực]
       │
       ▼
[GrayResize(size=84)] ───> [NGƯỜI 1 CÀI ĐẶT: RGB -> Grayscale Luminance, resize 84x84 uint8]
       │
       ▼
[FrameStackObservation(4)] (Chuẩn Gymnasium: Ghép thành tensor (4, 84, 84))
       │
       ▼
[Đầu vào ActorCritic]
```

### 1. Cài đặt lớp `FrameSkip(gym.Wrapper)`

- **Mục đích:** Mario đưa ra quyết định hành động ở tần số thấp hơn tần số khung hình gốc của NES (60 FPS). Lặp lại hành động `skip=4` lần giúp giảm 4 lần số bước tác tử cần học và tăng tốc độ thu thập dữ liệu lên gấp nhiều lần.
- **Đặc tả logic hàm `step(self, action)`:**
  1. Khởi tạo `total_reward = 0.0` (dạng `np.float32` có kích thước bằng với vector reward gốc, tức vector 5 phần tử).
  2. Tạo danh sách hoặc bộ đệm lưu trữ 2 khung hình gần nhất: `obs_buffer = []`.
  3. Lặp `for i in range(self.skip)`:
     - Gọi `obs, reward, terminated, truncated, info = self.env.step(action)`
     - Cộng dồn phần thưởng: `total_reward = total_reward + np.array(reward, dtype=np.float32)`
     - Đưa khung hình vào bộ đệm: `obs_buffer.append(obs)`. Nếu bộ đệm vượt quá 2 phần tử, loại bỏ phần tử cũ nhất (`obs_buffer = obs_buffer[-2:]`).
     - **Dừng sớm (Early exit):** Nếu `terminated or truncated` là `True`, lập tức `break` khỏi vòng lặp, không tiếp tục lặp các khung tiếp theo.
  4. Tính quan sát trả về:
     - Nếu bộ đệm có 2 khung: `max_frame = np.maximum(obs_buffer[0], obs_buffer[1])` (Lấy max theo từng điểm ảnh pixel-wise).
     - Nếu chỉ chạy được 1 khung hình (do kết thúc ngay khung đầu): `max_frame = obs_buffer[0]`.
  5. Trả về: `max_frame, total_reward, terminated, truncated, info`.

> **Ý nghĩa học thuật:** Máy chơi game NES thập niên 80 áp dụng cơ chế nhấp nháy sprite (sprite flickering) luân phiên hiển thị giữa các khung chẵn và lẻ để tránh tràn giới hạn phần cứng. Phép lấy `np.maximum` của 2 khung cuối đảm bảo không bao giờ bị mất dấu Mario hoặc quái vật Goomba.

---

### 2. Cài đặt lớp `GrayResize(gym.ObservationWrapper)`

- **Mục đích:** Giảm kích thước đầu vào mạng nơ-ron từ ảnh màu RGB $240 \times 256 \times 3$ xuống ảnh xám $84 \times 84 \times 1$ mà vẫn bảo toàn đầy đủ các đặc trưng cấu trúc địa hình và vật thể.
- **Đặc tả logic `__init__(self, env, size=84)`:**
  1. Gọi `super().__init__(env)`
  2. Lưu `self.size = size`
  3. Cập nhật lại không gian quan sát:
     ```python
     self.observation_space = gym.spaces.Box(
         low=0, high=255, shape=(size, size), dtype=np.uint8
     )
     ```
- **Đặc tả logic hàm `observation(self, obs)`:**
  1. Chuyển đổi ảnh RGB sang ảnh xám theo độ chói Luminance bằng OpenCV:
     ```python
     gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
     ```
     *(Tuyệt đối không lấy trung bình cộng `obs.mean(axis=-1)` vì công thức độ chói $Y = 0.299R + 0.587G + 0.114B$ phản ánh chính xác độ tương phản thực tế giữa Mario, kẻ thù và nền trời).*
  2. Resize ảnh xám về kích thước `(size, size)`:
     ```python
     resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
     ```
  3. Đảm bảo mảng trả về có kiểu dữ liệu `np.uint8` và shape `(size, size)`.

---

### 3. Cài đặt lớp `ScalarizeReward(gym.Wrapper)`

- **Mục đích:** Thuật toán PPO tiêu chuẩn tối ưu hóa một hàm phần thưởng vô hướng. Môi trường `mo-supermario-v0` trả về vector 5 mục tiêu $[x\_pos, time, death, coin, enemy]$. Wrapper này chuyển vector thành số thực thông qua tích vô hướng $w \cdot r$.
- **Đặc tả logic `__init__(self, env, weights)`:**
  1. Gọi `super().__init__(env)`
  2. Chuyển đổi weights thành mảng numpy float32: `self.weights = np.array(weights, dtype=np.float32)`
  3. Kiểm tra tính hợp lệ của số lượng trọng số:
     - Số lượng mục tiêu dự kiến: `num_objectives = env.unwrapped.reward_space.shape[0]` (nếu không có thì mặc định là 5).
     - Nếu `len(self.weights) != num_objectives`: Ném ngoại lệ `raise ValueError(f"weights count {len(self.weights)} != {num_objectives}")`.
- **Đặc tả logic hàm `step(self, action)`:**
  1. Gọi `obs, reward, terminated, truncated, info = self.env.step(action)`
  2. Sao lưu vector phần thưởng gốc:
     ```python
     info = dict(info)
     info["vector_reward"] = np.array(reward, dtype=np.float32)
     ```
  3. Tính tích vô hướng chuyển thành số thực chuẩn Python:
     ```python
     scalar_reward = float(np.dot(self.weights, reward))
     ```
  4. Trả về: `obs, scalar_reward, terminated, truncated, info`.

---

### 4. Triển khai Mẹo huấn luyện bổ sung (Tùy chọn ở cuối `wrappers.py`)

Viết thêm lớp `NoProgressTruncation(gym.Wrapper)` ở cuối tệp:
```python
class NoProgressTruncation(gym.Wrapper):
    """Cắt ngắn lượt chơi (truncated=True) nếu Mario không tiến lên phía trước
    quá max_steps bước tác tử liên tiếp. Giúp Mario thoát khỏi việc bị kẹt ở chân ống nước."""
    def __init__(self, env, max_steps=200):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps_without_progress = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.steps_without_progress = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_x = info.get("x_pos", 0)
        if curr_x > self.max_x + 2:
            self.max_x = curr_x
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
            if self.steps_without_progress >= self.max_steps:
                truncated = True
        return obs, reward, terminated, truncated, info
```

---

## III. QUY TRÌNH KIỂM THỬ VÀ LỆNH CHẠY CỦA NGƯỜI 1

Người 1 phải kiểm tra liên tục trên terminal:

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 1. Kiểm tra toàn bộ 7 bài test wrapper
pytest -q tests/test_wrappers.py

# 2. Kiểm tra chi tiết từng chức năng
pytest -q -k frameskip
pytest -q -k grayresize
pytest -q -k scalarize

# 3. Kiểm tra không gian quan sát khi ghép chuỗi
python -c "from mario_ppo.env import make_env; env = make_env('1-1'); obs, _ = env.reset(); print('Observation shape:', obs.shape, 'dtype:', obs.dtype)"
# Kết quả mong đợi: Observation shape: (4, 84, 84) dtype: uint8
```

---

## IV. LỘ TRÌNH 14 NGÀY CHI TIẾT CỦA NGƯỜI 1

- **Ngày 1:** Họp nhóm, rà soát API `gymnasium.Wrapper`, đọc kỹ `mario_ppo/wrappers.py` và `tests/test_wrappers.py`.
- **Ngày 2:** Viết phiên bản Alpha của `GrayResize` và `FrameSkip`.
- **Ngày 3:** Hoàn thiện `ScalarizeReward`. Chạy `pytest -q tests/test_wrappers.py` đạt **7/7 PASS**. Bàn giao cho Người 4.
- **Ngày 4:** Phối hợp Người 4 kiểm tra tính tương thích của wrappers với `make_env` trong `mario_ppo/env.py`.
- **Ngày 5:** Hỗ trợ kiểm thử pipeline toàn diện (`pytest -q` đạt 30/30 PASS).
- **Ngày 6:** Theo dõi tiến trình huấn luyện Màn 1-1, kiểm tra các khung hình hiển thị có bị lỗi sprite hay không.
- **Ngày 7:** Viết và kích hoạt `NoProgressTruncation` trong khối `if not eval_mode:` của `env.py` nếu Mario bị kẹt ống nước.
- **Ngày 8:** Kiểm tra các video `videos/step_*.mp4` của Màn 1-1 để đánh giá chất lượng tiền xử lý.
- **Ngày 9:** Cùng Người 4 kiểm tra kết quả đánh giá Màn 1-1 (`ket-qua-1-1.json`). Thống nhất chọn Màn thưởng 1-2.
- **Ngày 10:** Hỗ trợ Người 3 chuẩn bị môi trường cho Thí nghiệm so sánh đối chứng ($\gamma = 0.9$ vs $\gamma = 0.99$).
- **Ngày 11:** Phân tích thất bại (Failure Analysis): Phân loại nguyên nhân Mario chết (rơi hố, va chạm Goomba, hết giờ).
- **Ngày 12:** Hỗ trợ trích xuất video demo đẹp nhất của cả hai màn.
- **Ngày 13:** Viết Mục 1 (Mô tả Wrapper) và Mục 5 (Phân tích thất bại) trong Báo cáo học thuật 8 trang. Soạn thảo Slide 1–3.
- **Ngày 14:** Tham gia 2 buổi diễn tập thuyết trình và phản biện vấn đáp.

---

## V. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 1

1. **Hỏi:** *Tại sao trong FrameSkip phải lấy pixel-wise max của 2 khung cuối mà không lấy khung cuối đơn lẻ?*
   - **Đáp:** Máy NES có giới hạn phần cứng về số sprite trên một dòng quét nên áp dụng cơ chế nhấp nháy sprite (sprite flickering) xen kẽ giữa các khung chẵn và lẻ. Lấy max của 2 khung cuối đảm bảo mọi sprite (Mario, kẻ thù) đều được giữ lại trên quan sát đưa vào mạng nơ-ron.
2. **Hỏi:** *Tại sao GrayResize phải dùng cv2.COLOR_RGB2GRAY thay vì (R+G+B)/3?*
   - **Đáp:** Mắt người nhạy cảm với độ chói Luminance ($Y = 0.299R + 0.587G + 0.114B$). Trong Mario, nền trời màu xanh lam có giá trị B rất cao nhưng độ sáng thực tế lại tối hơn mặt đất xanh lục hoặc gạch đỏ. Lấy trung bình cộng sẽ làm mất độ tương phản giữa Mario và nền trời.
3. **Hỏi:** *Tại sao lại cần FrameStack 4 khung hình?*
   - **Đáp:** Một khung hình tĩnh chỉ cho biết vị trí (Position). Xếp chồng 4 khung hình liên tiếp cung cấp thông tin về vận tốc (Velocity) và gia tốc (Acceleration), giúp mạng biết Mario đang chạy hay đứng yên, đang nhảy lên hay rơi xuống.
4. **Hỏi:** *Sự khác nhau giữa terminated và truncated trong Gymnasium là gì?*
   - **Đáp:** `terminated` là kết thúc tự nhiên do luật chơi (chết, chạm cờ), tương lai bằng 0 và không bootstrap. `truncated` là cắt ngắn nhân tạo (hết giờ, cắt do kẹt), Mario vẫn còn sống và critic bắt buộc phải bootstrap từ $V(s_{final})$.
5. **Hỏi:** *Tại sao lại cần ScalarizeReward mà không huấn luyện trực tiếp đa mục tiêu?*
   - **Đáp:** Thuật toán PPO chuẩn tối ưu hàm mục tiêu vô hướng. Trọng số $w$ giúp định hướng tác tử ưu tiên tiến xa ($x\_pos$) và sợ chết ($death$), tránh bị phân tâm nhặt xu hay đạp quái.

---

## VI. CHECKLIST NGHIỆM THU CỦA NGƯỜI 1

- [ ] `FrameSkip` lặp đúng 4 lần, dừng khi terminated/truncated, cộng dồn vector reward.
- [ ] `FrameSkip` lấy pixel-wise max của 2 khung cuối cùng.
- [ ] `GrayResize` chuyển màu chuẩn Luminance và resize về đúng 84×84 `uint8`.
- [ ] `GrayResize` cập nhật `self.observation_space = gym.spaces.Box(0, 255, (84, 84), np.uint8)`.
- [ ] `ScalarizeReward` kiểm tra độ dài 5 trọng số, tính tích vô hướng $w \cdot r$ ra float.
- [ ] `ScalarizeReward` lưu vector gốc vào `info["vector_reward"]`.
- [ ] Chạy `pytest -q tests/test_wrappers.py` đạt **7/7 PASS**.
- [ ] Viết xong Mục 1 (Wrappers) và Mục 5 (Phân tích thất bại) cho Báo cáo học thuật.
- [ ] Hoàn thành Slide 1–3 và tự tin trả lời 5 câu hỏi vấn đáp chuyên sâu.
