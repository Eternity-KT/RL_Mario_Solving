# SỔ TAY KỸ THUẬT & PHÂN CÔNG TÁC CHIẾN 14 NGÀY - NGƯỜI 3
## VAI TRÒ: CHUYÊN GIA THUẬT TOÁN HỌC TĂNG CƯỜNG PPO & GAE (RL ALGORITHM & PPO/GAE LEAD)
### Đồ án: Dạy Super Mario Bros vượt màn bằng PPO (PyTorch) - Học phần Học tăng cường

---

## I. THÔNG TIN CHUNG & TRÁCH NHIỆM CỐT LÕI

- **Họ và tên thành viên:** [Điền tên Thành viên 3]
- **Vai trò chuyên trách:** Chuyên gia Thuật toán Học tăng cường, Lõi PPO & Ước lượng Lợi thế GAE.
- **Tệp mã nguồn sở hữu chính:** `mario_ppo/ppo.py` (Toàn quyền tác giả).
- **Tệp mã nguồn phối hợp:** `mario_ppo/model.py` (Phối hợp Người 2 về cấu trúc tensor gradient), `mario_ppo/train.py` (Phối hợp Người 4 về vòng lặp tối ưu hóa).
- **Bộ bài kiểm tra đơn vị cá nhân:** `tests/test_gae.py` (8 bài test) và `tests/test_ppo.py` (10 bài test) (Bắt buộc PASS 18/18 bài test).
- **Mục tiêu điểm số đóng góp:** Đạt trọn vẹn 18/18 điểm phần GAE & PPO trong Hạng mục A (Cài đặt thuật toán); chủ trì Mục 1 và Mục 4 (Ablation Study) trong Báo cáo học thuật (25 điểm).

---

## II. ĐẶC TẢ KỸ THUẬT 6 HÀM BẮT BUỘC TRONG `mario_ppo/ppo.py`

Người 3 chịu trách nhiệm hiện thực hóa "trái tim" toán học của toàn bộ dự án. Đây là tầng tính toán giải tích, ước lượng lợi thế và cập nhật gradient cho mạng nơ-ron.

```
[Dữ liệu Rollout: B = T * N = 256 * 16 = 4096 mẫu]
                         │
                         ▼
             [HÀM 1: compute_gae]
  - Tính sai số TD: delta_t = r_t + gamma * (1 - term) * V_next - values_t
  - Xử lý bootstrap khi truncated=1 (dùng final_values) và tại biên T-1
  - Truy hồi lùi: A_t = delta_t + gamma * lambda * (1 - term) * (1 - trunc) * A_{t+1}
  - Trả về: advantages và returns = advantages + values
                         │
                         ▼
             [HÀM 6: ppo_update]
  - Lặp qua update_epochs = 4; xáo trộn ngẫu nhiên mẫu (torch.randperm)
  - Chia thành num_minibatches = 8 (512 mẫu / mini-batch)
  - Mỗi mini-batch: model(obs) -> logits, values -> Categorical -> new_logp, entropy
                         │
                         ▼
             [HÀM 5: ppo_loss]
  - Chuẩn hóa lợi thế trên mini-batch: (A - mean) / (std + 1e-8) với unbiased=False
  - Gọi [HÀM 2: clipped_policy_loss]: L_clip = -mean(min(r * A, clip(r) * A))
  - Gọi [HÀM 3: value_loss]: L_vf = mean((V - R)^2) (+ clipped value loss)
  - Gọi [HÀM 4: approx_kl_and_clipfrac]: k3 = mean((r - 1) - log r)
  - Tổng hợp: Loss = L_clip + vf_coef * L_vf - ent_coef * Entropy
                         │
                         ▼
  - loss.backward() -> nn.utils.clip_grad_norm_(0.5) -> optimizer.step()
  - Dừng sớm (early stop) nếu approx_kl > target_kl
```

---

### 1. Hàm `compute_gae(...)`

```python
def compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma, lam):
```

- **Mục đích:** Ước lượng lợi thế tổng quát GAE ($\hat{A}_t$) và mục tiêu của Critic ($R_t$).
- **Đầu vào (Tensor float 2 chiều shape `(T, N)`):**
  - `rewards[t, n]`: Phần thưởng nhận được sau hành động ở bước $t$.
  - `values[t, n]`: Giá trị trạng thái $V(S_t)$ do Critic dự đoán lúc thu thập dữ liệu.
  - `terminated[t, n]`: 1.0 nếu lượt KẾT THÚC thật (Mario chết, chạm cờ); 0.0 nếu chưa kết thúc.
  - `truncated[t, n]`: 1.0 nếu lượt BỊ CẮT ngắn (hết giờ, cắt do kẹt); 0.0 nếu không.
  - `final_values[t, n]`: $V(s_{final})$ của quan sát cuối khi bị cắt; chỉ có nghĩa ở vị trí `truncated = 1`.
  - `last_value[n]`: Giá trị trạng thái $V(S_T)$ ngay sau bước cuối cùng của đoạn rollout.
  - `gamma, lam`: Hệ số chiết khấu ($\gamma = 0.9$) và tham số GAE ($\lambda = 0.95$).
- **Đặc tả thuật toán:**
  1. Khởi tạo `advantages = torch.zeros_like(rewards)`.
  2. Khởi tạo `last_gae_lam = 0.0` (shape `(N,)`).
  3. Duyệt ngược thời gian từ $t = T - 1$ lùi về $0$:
     - Xác định giá trị kế tiếp $V_{next}$:
       + Nếu $t == T - 1$: Nếu bước cuối chưa kết thúc (`terminated == 0`) và chưa bị cắt (`truncated == 0`), coi như bị cắt với bootstrap $V_{next} = last\_value$. Nếu bị cắt thì $V_{next} = final\_values[t]$. Nếu kết thúc thì $V_{next} = 0.0$.
       + Nếu $t < T - 1$: Nếu bước $t$ bị cắt (`truncated[t] == 1`), $V_{next} = final\_values[t]$; ngược lại $V_{next} = values[t + 1]$.
     - Tính sai số thời gian sai biệt (TD Error):
       $$\delta_t = rewards[t] + \gamma \cdot (1 - terminated[t]) \cdot V_{next} - values[t]$$
     - Tính lợi thế GAE truy hồi:
       $$A_t = \delta_t + \gamma \cdot \lambda \cdot (1 - terminated[t]) \cdot (1 - truncated[t]) \cdot last\_gae\_lam$$
     - Cập nhật `advantages[t] = A_t` và `last_gae_lam = A_t`.
  4. Tính mục tiêu của Critic: `returns = advantages + values`.
  5. Trả về: `(advantages, returns)`.

---

### 2. Hàm `clipped_policy_loss(ratio, advantages, clip)`

- **Công thức:**
  $$L_{clip} = -\frac{1}{B} \sum_{i=1}^B \min\left( r_i \cdot \hat{A}_i, \;\text{clamp}(r_i, 1 - \epsilon, 1 + \epsilon) \cdot \hat{A}_i \right)$$
- **Đặc tả logic:**
  1. `pg_loss1 = ratio * advantages`
  2. `pg_loss2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages`
  3. `loss = -torch.min(pg_loss1, pg_loss2).mean()`
  4. Trả về `loss` dạng tensor vô hướng có gradient.

---

### 3. Hàm `value_loss(values, returns, old_values=None, clip=None)`

- **Công thức:**
  - Mất mát MSE không cắt: $L_{vf} = \text{mean}\left( (values - returns)^2 \right)$ *(Lưu ý: Không có hệ số 1/2)*.
  - Biến thể cắt (khi `old_values` và `clip` khác `None`):
    $$v_{clip} = old\_values + \text{clamp}(values - old\_values, -clip, clip)$$
    $$L_{vf} = \text{mean}\left( \max\left( (values - returns)^2, \; (v_{clip} - returns)^2 \right) \right)$$
- **Đặc tả logic:**
  1. `v_loss_unclipped = (values - returns) ** 2`
  2. Nếu `old_values is not None and clip is not None`:
     `v_clipped = old_values + torch.clamp(values - old_values, -clip, clip)`
     `v_loss_clipped = (v_clipped - returns) ** 2`
     `return torch.max(v_loss_unclipped, v_loss_clipped).mean()`
  3. Ngược lại: `return v_loss_unclipped.mean()`

---

### 4. Hàm `approx_kl_and_clipfrac(log_ratio, clip)`

- **Công thức:**
  - $ratio = \exp(log\_ratio)$
  - Ước lượng KL $k_3$: $approx\_kl = \text{mean}\left( (ratio - 1) - log\_ratio \right)$
  - Tỷ lệ cắt: $clipfrac = \text{mean}\left( |ratio - 1| > clip \right)$
- **Đặc tả logic:**
  1. `with torch.no_grad():`
  2. `ratio = torch.exp(log_ratio)`
  3. `approx_kl = ((ratio - 1.0) - log_ratio).mean().item()`
  4. `clipfrac = ((ratio - 1.0).abs() > clip).float().mean().item()`
  5. Trả về: `(float(approx_kl), float(clipfrac))`.

---

### 5. Hàm `ppo_loss(new_logp, entropy, values, batch, cfg)`

- **Đầu vào:**
  - `new_logp, entropy, values`: Tính từ mạng hiện tại trên mini-batch (có gradient).
  - `batch`: Dictionary chứa `old_logp, advantages, returns, old_values` (cố định).
  - `cfg`: Cấu hình chứa `clip_coef, vf_coef, ent_coef, norm_adv, clip_vloss`.
- **Đặc tả logic:**
  1. Tính `log_ratio = new_logp - batch["old_logp"]` và `ratio = torch.exp(log_ratio)`.
  2. Chuẩn hóa lợi thế nếu `cfg.norm_adv`:
     ```python
     adv = batch["advantages"]
     if cfg.norm_adv:
         std = torch.std(adv, unbiased=False)
         adv = (adv - adv.mean()) / (std + 1e-8)
     ```
  3. Gọi `pg_loss = clipped_policy_loss(ratio, adv, cfg.clip_coef)`.
  4. Gọi `v_loss = value_loss(values, batch["returns"], batch["old_values"] if cfg.clip_vloss else None, cfg.clip_coef if cfg.clip_vloss else None)`.
  5. Tính `entropy_loss = entropy.mean()`.
  6. Tổng hợp mất mát:
     ```python
     total_loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss
     ```
  7. Gọi `approx_kl, clipfrac = approx_kl_and_clipfrac(log_ratio, cfg.clip_coef)`.
  8. Trả về `total_loss` và dictionary `stats`:
     ```python
     stats = {
         "policy_loss": pg_loss.item(),
         "value_loss": v_loss.item(),
         "entropy": entropy_loss.item(),
         "approx_kl": approx_kl,
         "clipfrac": clipfrac
     }
     return total_loss, stats
     ```

---

### 6. Hàm `ppo_update(model, optimizer, batch, cfg)`

- **Đặc tả logic vòng lặp tối ưu hóa:**
  1. Lấy kích thước batch $B = len(batch["obs"])$.
  2. Tính kích thước mini-batch: `batch_size = B // cfg.num_minibatches`.
  3. Khởi tạo danh sách thống kê: `all_stats = []`, `minibatch_updates = 0`.
  4. Lặp `for epoch in range(cfg.update_epochs)`:
     - Tạo hoán vị ngẫu nhiên chỉ số: `indices = torch.randperm(B)`.
     - Lặp qua từng mini-batch:
       ```python
       for start in range(0, B, batch_size):
           minibatch_updates += 1
           mb_idx = indices[start : start + batch_size]
           mb = {k: v[mb_idx] for k, v in batch.items()}
           
           # Forward mạng nơ-ron
           logits, values = model(mb["obs"])
           dist = torch.distributions.Categorical(logits=logits)
           new_logp = dist.log_prob(mb["actions"])
           entropy = dist.entropy()
           
           # Tính loss và cập nhật
           loss, stats = ppo_loss(new_logp, entropy, values, mb, cfg)
           optimizer.zero_grad()
           loss.backward()
           nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
           optimizer.step()
           all_stats.append(stats)
     ```
     - **Kiểm tra dừng sớm (Early Stopping):**
       Nếu `cfg.target_kl is not None`:
       Tính `mean_kl = np.mean([s["approx_kl"] for s in all_stats[-cfg.num_minibatches:]])`.
       Nếu `mean_kl > cfg.target_kl`: Dừng vòng lặp các epoch còn lại (`break`).
  5. Tính trung bình các đại lượng trong `all_stats`, thêm `"minibatch_updates": minibatch_updates`.
  6. Trả về dictionary thống kê trung bình.

---

## III. QUY TRÌNH KIỂM THỬ VÀ LỆNH CHẠY CỦA NGƯỜI 3

Người 3 thực hiện kiểm thử tự động trên terminal:

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 1. Kiểm tra 8 bài test GAE
pytest -q tests/test_gae.py

# 2. Kiểm tra 10 bài test PPO
pytest -q tests/test_ppo.py

# 3. Kiểm tra các trường hợp biên đặc thù
pytest -q -k bootstrap          # Kiểm tra xử lý cắt ngắn và biên rollout
pytest -q -k kl_early_stop      # Kiểm tra cơ chế dừng sớm theo KL
pytest -q -k normalised         # Kiểm tra chuẩn hóa lợi thế unbiased=False
```

---

## IV. LỘ TRÌNH 14 NGÀY CHI TIẾT CỦA NGƯỜI 3

- **Ngày 1:** Đọc hiểu công thức toán học GAE và PPO, đối chiếu với `tests/test_gae.py` và `tests/test_ppo.py`.
- **Ngày 2:** Viết phiên bản Alpha của hàm `compute_gae`, xử lý kỹ trường hợp biên $T-1$ và $truncated=1$.
- **Ngày 3:** Hoàn thiện `compute_gae`, `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`. Chạy `pytest -q tests/test_gae.py` đạt **8/8 PASS**.
- **Ngày 4:** Viết hoàn chỉnh `ppo_loss` và `ppo_update`. Chạy `pytest -q tests/test_ppo.py` đạt **10/10 PASS**. Bàn giao toàn bộ lõi PPO cho Người 4.
- **Ngày 5:** Hỗ trợ Người 4 chạy smoke test tích hợp. Đảm bảo toàn bộ 30 bài kiểm tra công khai PASS 100%.
- **Ngày 6:** Theo dõi chỉ số `approx_kl` và `clipfrac` trong phiên huấn luyện chính thức Màn 1-1.
- **Ngày 7:** Cùng Người 4 chẩn đoán đồ thị learning curve. Điều chỉnh `ent_coef` nếu Mario thiếu tính khám phá.
- **Ngày 8:** Tổng kết số liệu Run 1 Màn 1-1. Thiết kế chi tiết cấu hình Thí nghiệm so sánh đối chứng (Ablation Study).
- **Ngày 9:** Cùng Người 4 xác nhận checkpoint 1-1 đạt `flag_rate >= 0.8`. Bắt đầu chạy thí nghiệm đối chứng.
- **Ngày 10 (Chủ trì Ablation Study):** Chạy tiến trình đối chứng $\gamma = 0.99$ (so với Baseline $\gamma = 0.9$) trên Màn 1-1 cùng ngân sách 2M bước và cùng seed=1:
  ```powershell
  python -m mario_ppo.train level=1-1 algo.gamma=0.99 algo.total_steps=2000000 run_dir=runs/ablation_gamma_099 seed=1
  ```
- **Ngày 11:** Thu thập dữ liệu từ `progress.csv` của 2 lần chạy, trích xuất biểu đồ so sánh lợi tức và tốc độ hội tụ.
- **Ngày 12:** Viết nhận xét phân tích học thuật sâu sắc về tác động của $\gamma$ đến phương sai của Critic và độ ổn định của Actor.
- **Ngày 13:** Viết Mục 1 (Mô tả GAE & PPO) và chủ trì toàn bộ Mục 4 (Thí nghiệm so sánh đối chứng) trong Báo cáo 8 trang. Soạn Slide 6–7.
- **Ngày 14:** Tham gia 2 buổi diễn tập thuyết trình và phản biện vấn đáp.

---

## V. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 3

1. **Hỏi:** *GAE (Generalized Advantage Estimation) giải quyết sự đánh đổi giữa Bias và Variance như thế nào?*
   - **Đáp:** Monte Carlo ($\lambda=1$) không chệch (unbiased) nhưng phương sai cực lớn do tích lũy nhiễu ngẫu nhiên của cả episode. TD(0) ($\lambda=0$) phương sai rất nhỏ nhưng chệch (bias) nặng do phụ thuộc vào giá trị Critic ban đầu. GAE kết hợp hai thái cực qua trung bình trọng số hàm mũ của các bước TD k-step với $\lambda \in [0, 1]$. Giá trị $\lambda=0.95$ giữ phương sai đủ nhỏ để gradient ổn định mà bias vẫn ở mức kiểm soát được.
2. **Hỏi:** *Bản chất toán học của Clipped Surrogate Objective trong PPO là gì?*
   - **Đáp:** Tỷ lệ $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{old}(a_t|s_t)$. Mục tiêu cắt lấy $\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$. Khi $A_t > 0$, không cho $r_t$ tăng quá $1+\epsilon$; khi $A_t < 0$, không cho $r_t$ giảm quá $1-\epsilon$. Điều này tạo ra một cận dưới bi quan (Pessimistic Lower Bound), ngăn chính sách mới lệch quá xa chính sách cũ mà không cần tính ma trận nghịch đảo Fisher phức tạp như TRPO.
3. **Hỏi:** *Tại sao công thức approx_kl = mean((ratio - 1) - log_ratio) ưu việt hơn -mean(log_ratio)?*
   - **Đáp:** Đây là ước lượng $k_3$ của John Schulman. Công thức cũ $-mean(\log r)$ có thể nhận giá trị âm do nhiễu mẫu. Ước lượng $k_3$ dựa trên bất đẳng thức $(x - 1) - \ln x \ge 0$ với mọi $x > 0$. Vì $ratio > 0$, $k_3$ luôn đảm bảo không âm, đo lường chính xác khoảng cách phân kỳ KL thực tế giữa hai phân phối và rất ổn định.
4. **Hỏi:** *Tại sao phải chuẩn hóa Advantage trên từng mini-batch trong ppo_loss?*
   - **Đáp:** Quy mô phần thưởng và lợi thế thay đổi mạnh giữa các giai đoạn (đầu game thấp, cuối game cao khi sắp chạm cờ). Nếu không chuẩn hóa, độ lớn gradient dao động dữ dội. Phép chuẩn hóa đưa lợi thế về kỳ vọng 0 và phương sai 1, giúp bước nhảy gradient luôn ổn định, giảm phương sai và tăng tốc hội tụ.
5. **Hỏi:** *Tại sao trong ppo.py lại dùng torch.std(advantages, unbiased=False) thay vì unbiased=True?*
   - **Đáp:** `unbiased=True` dùng hiệu chỉnh Bessel (chia cho $N-1$). Trong mã nguồn chuẩn PPO (OpenAI Baselines, CleanRL), độ lệch chuẩn được tính trên toàn bộ mẫu trong mini-batch (chia cho $N$, tương ứng `unbiased=False`). Tuân thủ `unbiased=False` đảm bảo tương thích tuyệt đối với bài test công khai `tests/test_ppo.py`.

---

## VI. CHECKLIST NGHIỆM THU CỦA NGƯỜI 3

- [ ] `compute_gae` tính đúng sai số TD $\delta_t$ và bootstrap từ $final\_values$ khi $truncated=1$.
- [ ] `compute_gae` xử lý đúng biên $T-1$ với bootstrap $last\_value$.
- [ ] `clipped_policy_loss` tính đúng $-\text{mean}(\min(rA, \text{clip}(r)A))$.
- [ ] `value_loss` tính đúng MSE $\text{mean}((V - R)^2)$ và clipped variant.
- [ ] `approx_kl_and_clipfrac` tính đúng chỉ số $k_3$ không âm và tỷ lệ cắt `clipfrac`.
- [ ] `ppo_loss` chuẩn hóa advantage với `unbiased=False` và tổng hợp đủ 3 thành phần loss.
- [ ] `ppo_update` thực hiện đúng epochs, minibatches, cắt chuẩn gradient 0.5 và KL early stop.
- [ ] Chạy `pytest -q tests/test_gae.py` đạt **8/8 PASS** và `tests/test_ppo.py` đạt **10/10 PASS**.
- [ ] Hoàn thành Thí nghiệm đối chứng $\gamma=0.9$ vs $\gamma=0.99$ trên Màn 1-1 (2M bước, seed=1).
- [ ] Viết xong Mục 1 (GAE & PPO) và Mục 4 (Ablation Study) cho Báo cáo học thuật.
- [ ] Hoàn thành Slide 6–7 và tự tin trả lời 5 câu hỏi vấn đáp chuyên sâu.
