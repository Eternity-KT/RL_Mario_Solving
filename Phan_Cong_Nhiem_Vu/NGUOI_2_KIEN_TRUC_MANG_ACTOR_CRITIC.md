# SỔ TAY KỸ THUẬT & PHÂN CÔNG TÁC CHIẾN 14 NGÀY - NGƯỜI 2
## VAI TRÒ: CHUYÊN GIA KIẾN TRÚC MẠNG ACTOR-CRITIC (DEEP LEARNING & ACTOR-CRITIC LEAD)
### Đồ án: Dạy Super Mario Bros vượt màn bằng PPO (PyTorch) - Học phần Học tăng cường

---

## I. THÔNG TIN CHUNG & TRÁCH NHIỆM CỐT LÕI

- **Họ và tên thành viên:** [Điền tên Thành viên 2]
- **Vai trò chuyên trách:** Chuyên gia Kiến trúc Mạng nơ-ron Actor-Critic & Học sâu.
- **Tệp mã nguồn sở hữu chính:** `mario_ppo/model.py` (Toàn quyền tác giả).
- **Tệp mã nguồn phối hợp:** `mario_ppo/ppo.py` (Phối hợp Người 3 về giao diện logits và value trong mini-batch), `mario_ppo/train.py` (Phối hợp Người 4 về lưu/tải checkpoint).
- **Bộ bài kiểm tra đơn vị cá nhân:** `tests/test_model.py` (Bắt buộc PASS 5/5 bài test).
- **Mục tiêu điểm số đóng góp:** Đạt trọn vẹn 6/6 điểm phần Model trong Hạng mục A (Cài đặt thuật toán); đóng góp Mục 1 và Mục 2 trong Báo cáo học thuật (25 điểm).

---

## II. ĐẶC TẢ KỸ THUẬT KIẾN TRÚC MẠNG TRONG `mario_ppo/model.py`

Người 2 chịu trách nhiệm thiết kế và hiện thực hóa lớp `ActorCritic(nn.Module)`. Mô hình đóng vai trò kép: vừa là **Actor** (quyết định hành động thông qua logits) vừa là **Critic** (định giá trạng thái để tính toán lợi thế GAE).

```
          [Quan sát: Tensor uint8 (B, 4, 84, 84)]
                            │
                            ▼
     [Chuẩn hóa điểm ảnh: x = obs.float() / 255.0]
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │           THÂN MẠNG TÍCH CHẬP CHUNG          │
     │                 (CNN BACKBONE)               │
     │                                              │
     │ Conv2D 1: in=4,  out=32, k=8, s=4, p=0 (ReLU)│ ──> (B, 32, 20, 20)
     │ Conv2D 2: in=32, out=64, k=4, s=2, p=0 (ReLU)│ ──> (B, 64, 9, 9)
     │ Conv2D 3: in=64, out=64, k=3, s=1, p=0 (ReLU)│ ──> (B, 64, 7, 7)
     │ Flatten:  64 * 7 * 7 = 3136 phần tử          │ ──> (B, 3136)
     │ Linear:   in=3136, out=512             (ReLU)│ ──> (B, 512)
     └──────────────────────┬───────────────────────┘
                            │ (Latent Features: 512)
             ┌──────────────┴──────────────┐
             ▼                             ▼
    [ACTOR / POLICY HEAD]         [CRITIC / VALUE HEAD]
    Linear(512, n_actions)        Linear(512, 1) + squeeze(-1)
    gain = 0.01, bias = 0         gain = 1.0,  bias = 0
             │                             │
             ▼                             ▼
    Logits: (B, n_actions)        State Value: (B,)
    float32                       float32
```

---

### 1. Đặc tả logic hàm khởi tạo `__init__(self, n_actions, in_channels=4)`

Hàm `__init__` phải khởi tạo 3 thành phần chính:
1. **Thân CNN trích xuất đặc trưng (Shared CNN Trunk):**
   ```python
   self.trunk = nn.Sequential(
       _init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), gain=np.sqrt(2)),
       nn.ReLU(),
       _init(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=np.sqrt(2)),
       nn.ReLU(),
       _init(nn.Conv2d(64, 64, kernel_size=3, stride=1), gain=np.sqrt(2)),
       nn.ReLU(),
       nn.Flatten(),
       _init(nn.Linear(64 * 7 * 7, 512), gain=np.sqrt(2)),
       nn.ReLU(),
   )
   ```
2. **Đầu ra Chính sách (Actor Policy Head):**
   ```python
   self.policy = _init(nn.Linear(512, n_actions), gain=0.01)
   ```
   > **Lưu ý sống còn:** Hệ số `gain=0.01` là bắt buộc theo quy chuẩn PPO. Nó giúp các logits ban đầu rất nhỏ xấp xỉ 0, khiến xác suất phân phối hành động qua Softmax ban đầu gần như phân bố đều (uniform distribution). Mario sẽ thử nghiệm tất cả 7 phím bấm với cơ hội như nhau, đạt entropy cực đại $\ln(7) \approx 1.94$, ngăn chặn hiện tượng Mario bị kẹt vào một nút phím duy nhất ngay khi vừa khởi tạo.
3. **Đầu ra Giá trị trạng thái (Critic Value Head):**
   ```python
   self.value = _init(nn.Linear(512, 1), gain=1.0)
   ```
   > Hệ số `gain=1.0` giúp mạng Critic có độ biến thiên phù hợp ngay từ đầu để học bài toán hồi quy giá trị trạng thái.

Hàm trợ giúp `_init(layer, gain)` đã được định nghĩa sẵn ở đầu tệp `mario_ppo/model.py`:
```python
def _init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer
```

---

### 2. Đặc tả logic hàm lan truyền xuôi `forward(self, obs)`

- **Đầu vào:** `obs` dạng tensor PyTorch kiểu `torch.uint8` có shape `(B, C, 84, 84)` nằm trên thiết bị CPU hoặc CUDA.
- **Các bước tính toán:**
  1. Ép kiểu và chuẩn hóa điểm ảnh về khoảng $[0.0, 1.0]$:
     ```python
     x = obs.float() / 255.0
     ```
  2. Truyền qua thân CNN dùng chung để lấy vector đặc trưng 512 chiều:
     ```python
     features = self.trunk(x)
     ```
  3. Tính toán logits của chính sách:
     ```python
     logits = self.policy(features)  # Shape: (B, n_actions), kiểu float32
     ```
  4. Tính toán giá trị trạng thái của critic và ép phẳng chiều cuối:
     ```python
     value = self.value(features).squeeze(-1)  # Shape: (B,), kiểu float32
     ```
  5. Trả về: `(logits, value)`.

---

### 3. Kiểm tra tính tương thích với hàm `act(self, obs)`

Phương thức `act` đã có sẵn trong `model.py` để phục vụ thu thập dữ liệu trong `train.py` và `evaluate.py`:
```python
@torch.no_grad()
def act(self, obs):
    device = next(self.parameters()).device
    logits, value = self(obs.to(device))
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action.cpu(), dist.log_prob(action).cpu(), value.cpu()
```
Người 2 phải kiểm tra:
- `logits` trả về từ `forward` phải khớp với số hành động `n_actions`.
- `value` trả về phải có shape `(B,)` để `dist.sample()` và `dist.log_prob(action)` có cùng shape với `value`.
- Khi gọi `act(obs)`, toàn bộ tensor trả về phải nằm trên CPU và không chứa đồ thị đạo hàm (`requires_grad=False`).

---

## III. QUY TRÌNH KIỂM THỬ VÀ LỆNH CHẠY CỦA NGƯỜI 2

Người 2 thực hiện kiểm thử tự động trên terminal:

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 1. Kiểm tra toàn bộ 5 bài test của model.py
pytest -q tests/test_model.py

# 2. Kiểm tra chi tiết từng tính chất
pytest -q -k forward_shapes       # Kiểm tra shape (B, n_actions) và (B,)
pytest -q -k uniform              # Kiểm tra chính sách ban đầu gần đều
pytest -q -k gradients            # Kiểm tra gradient lan truyền đến 100% tham số
pytest -q -k act                  # Kiểm tra act lấy mẫu Categorical hợp lệ

# 3. Kiểm tra số lượng tham số mô hình
python -c "from mario_ppo.model import ActorCritic; m = ActorCritic(7); print('Tổng tham số:', sum(p.numel() for p in m.parameters()))"
# Kết quả mong đợi: ~1.8 triệu tham số (tương đương chuẩn Nature CNN)
```

---

## IV. LỘ TRÌNH 14 NGÀY CHI TIẾT CỦA NGƯỜI 2

- **Ngày 1:** Đọc hiểu kiến trúc Nature CNN, nghiên cứu khởi tạo trực giao `orthogonal_`, đọc kỹ `mario_ppo/model.py` và `tests/test_model.py`.
- **Ngày 2:** Viết cấu trúc `__init__` với CNN 3 tầng tích chập, kiểm tra kích thước output feature map $7 \times 7$.
- **Ngày 3:** Hoàn thiện `forward` và khởi tạo trực giao chuẩn xác (`gain=0.01` cho policy). Chạy `pytest -q tests/test_model.py` đạt **5/5 PASS**. Bàn giao cho Người 3 và Người 4.
- **Ngày 4:** Phối hợp Người 3 kiểm tra truyền nhận tensor trong mini-batch của `ppo_update`. Sửa các lỗi tensor shape nếu có.
- **Ngày 5:** Hỗ trợ Người 4 chạy smoke test 10,000 bước. Khảo sát mức tiêu thụ RAM của mô hình.
- **Ngày 6:** Viết script trích xuất dữ liệu từ `progress.csv` và vẽ biểu đồ động học của mô hình.
- **Ngày 7:** Theo dõi chỉ số `entropy` và `clipfrac`. Kiểm tra xem phân phối hành động có bị co cụm quá sớm hay không.
- **Ngày 8:** Tổng hợp dữ liệu Màn 1-1; xuất 6 đồ thị đường cong học tập sơ bộ.
- **Ngày 9:** Cùng Người 4 kiểm tra checkpoint 1-1 (`flag_rate >= 0.8`). Chuẩn bị mạng cho Transfer Learning Màn thưởng 1-2.
- **Ngày 10:** Hỗ trợ Người 3 và Người 4 chạy song song Màn thưởng và Thí nghiệm đối chứng gamma.
- **Ngày 11:** Vẽ biểu đồ so sánh loss của Critic và Actor giữa 2 cấu hình đối chứng.
- **Ngày 12:** Xuất bản trọn bộ 7 biểu đồ phân tích đạt chuẩn 300 DPI cho Báo cáo học thuật.
- **Ngày 13:** Viết Mục 1 (Kiến trúc Actor-Critic) và Mục 2 (Cấu hình phần cứng & SPS) trong Báo cáo 8 trang. Soạn thảo Slide 4–5.
- **Ngày 14:** Tham gia 2 buổi tổng duyệt thuyết trình và trả lời vấn đáp thử nghiệm.

---

## V. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 2

1. **Hỏi:** *Tại sao lại dùng chung thân CNN (Shared Backbone) cho cả Actor và Critic thay vì 2 mạng riêng biệt?*
   - **Đáp:** Dùng chung CNN giúp tiết kiệm 50% thời gian tính toán và bộ nhớ RAM (yếu tố sống còn khi chạy CPU). Đồng thời, việc Critic học cách đánh giá trạng thái tốt/xấu buộc thân CNN phải học các biểu diễn thị giác không gian chất lượng cao, trực tiếp hỗ trợ Actor ra quyết định sắc bén hơn.
2. **Hỏi:** *Tại sao Policy Head bắt buộc khởi tạo trực giao với gain = 0.01?*
   - **Đáp:** `gain=0.01` làm cho trọng số và bias ban đầu rất gần 0. Khi qua hàm Linear, các logits của 7 hành động xấp xỉ bằng nhau. Qua phân phối Softmax, xác suất chọn mỗi hành động xấp xỉ $1/7 \approx 14.3\%$, entropy ban đầu đạt cực đại xấp xỉ $\ln(7) \approx 1.94$. Điều này kích thích Mario khám phá toàn diện mọi phím bấm, không bị thiên vị một phím nào lúc khởi tạo.
3. **Hỏi:** *Tại sao Critic Head lại khởi tạo với gain = 1.0?*
   - **Đáp:** Critic Head thực hiện bài toán hồi quy giá trị $V(s)$ đại diện cho tổng phần thưởng chiết khấu kỳ vọng (biên độ lớn hơn logits rất nhiều). Khởi tạo `gain=1.0` giúp giữ nguyên phương sai tín hiệu qua các tầng, giúp Critic nhanh chóng bắt kịp quy mô lợi tức thực tế của game.
4. **Hỏi:** *Tại sao đầu vào obs uint8 [0, 255] phải chia cho 255.0 về [0.0, 1.0]?*
   - **Đáp:** Nếu để nguyên [0, 255], các phép nhân tích chập Conv2D sẽ khiến giá trị kích hoạt bùng nổ lên hàng ngàn, làm bão hòa các hàm kích hoạt phi tuyến và gây bùng nổ gradient. Chuẩn hóa về [0, 1] giúp gradient ổn định và mạng hội tụ nhanh.
5. **Hỏi:** *Tại sao hàm act(obs) lại sử dụng decorator @torch.no_grad()?*
   - **Đáp:** Giai đoạn Rollout chỉ cần suy luận xuôi (forward) để lấy mẫu hành động tương tác với game. `@torch.no_grad()` ngăn PyTorch xây dựng đồ thị tính toán đạo hàm, tiết kiệm hàng gigabyte bộ nhớ RAM và tăng tốc độ thu dữ liệu lên gấp đôi. Đạo hàm chỉ được tính toán trong vòng lặp cập nhật `ppo_update`.

---

## VI. CHECKLIST NGHIỆM THU CỦA NGƯỜI 2

- [ ] `ActorCritic.__init__`: CNN 3 tầng tích chập đúng kernel, stride và Linear 512.
- [ ] Khởi tạo trực giao: Conv/Linear ẩn `gain=sqrt(2)`, Policy `gain=0.01`, Critic `gain=1.0`, bias=0.
- [ ] `ActorCritic.forward`: Chuẩn hóa `obs.float() / 255.0`.
- [ ] `ActorCritic.forward`: Policy trả về logits shape `(B, n_actions)` float32.
- [ ] `ActorCritic.forward`: Critic trả về value shape `(B,)` float32 đã squeeze.
- [ ] `act(obs)` lấy mẫu Categorical hợp lệ, không đạo hàm, trả tensor CPU.
- [ ] Chạy `pytest -q tests/test_model.py` đạt **5/5 PASS**.
- [ ] Viết xong Mục 1 (Kiến trúc mạng) và Mục 2 (Cấu hình phần cứng & SPS) trong Báo cáo.
- [ ] Hoàn thành Slide 4–5 và tự tin trả lời 5 câu hỏi vấn đáp chuyên sâu.
