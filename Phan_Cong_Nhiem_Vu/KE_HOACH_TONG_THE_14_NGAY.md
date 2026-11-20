# KẾ HOẠCH TỔNG THỂ & QUẢN TRỊ DỰ ÁN HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)
## ĐỀ TÀI: DẠY SUPER MARIO BROS VƯỢT MÀN BẰNG THUẬT TOÁN PROXIMAL POLICY OPTIMIZATION (PPO) TỪ ĐIỂM ẢNH
### SPRINT TÁC CHIẾN 14 NGÀY CHO NHÓM 04 SINH VIÊN (HỌC KỲ 1, NĂM HỌC 2026 – 2027)

---

> **Tài liệu Word chính thức:** [`Phan_cong_cong_viec_nhom_Mario_PPO_4_nguoi_14_ngay.docx`](../Phan_cong_cong_viec_nhom_Mario_PPO_4_nguoi_14_ngay.docx) *(Đã xuất bản tại thư mục gốc của dự án, định dạng A4 chuẩn in ấn học thuật).*

---

## MỤC LỤC TỔNG QUAN

1. [Tổng quan dự án và Đối chiếu yêu cầu đề bài chính thức](#1-tổng-quan-dự-án-và-đối-chiếu-yêu-cầu-đề-bài-chính-thức)
2. [Kiến trúc kỹ thuật và Luồng dữ liệu toàn hệ thống](#2-kiến-trúc-kỹ-thuật-và-luồng-dữ-liệu-toàn-hệ-thống)
3. [Phân chia vai trò 4 thành viên & Ma trận RACI](#3-phân-chia-vai-trò-4-thành-viên--ma-trận-raci)
4. [Lộ trình tác chiến chi tiết 14 ngày (Day-by-Day Timeline)](#4-lộ-trình-tác-chiến-chi-tiết-14-ngày-day-by-day-timeline)
5. [Chiến lược huấn luyện, Tài nguyên tính toán và Đánh giá định lượng](#5-chiến-lược-huấn-luyện-tài-nguyên-tính-toán-và-đánh-giá-định-lượng)
6. [Quy chuẩn Báo cáo học thuật 8 trang, Slide thuyết trình và Ngân hàng Vấn đáp](#6-quy-chuẩn-báo-cáo-học-thuật-8-trang-slide-thuyết-trình-và-ngân-hàng-vấn-đáp)
7. [Quản trị mã nguồn, Tính tái lập và Đóng gói hồ sơ nộp bài](#7-quản-trị-mã-nguồn-tính-tái-lập-và-đóng-gói-hồ-sơ-nộp-bài)
8. [Bảng quản trị 15 rủi ro kỹ thuật và Kế hoạch ứng phó](#8-bảng-quản-trị-15-rủi-ro-kỹ-thuật-và-kế-hoạch-ứng-phó)
9. [Bộ tiêu chí nghiệm thu và Checklist toàn diện](#9-bộ-tiêu-chí-nghiệm-thu-và-checklist-toàn-diện)

---

## 1. TỔNG QUAN DỰ ÁN VÀ ĐỐI CHIẾU YÊU CẦU ĐỀ BÀI CHÍNH THỨC

### 1.1. Bối cảnh và 4 Mục tiêu bắt buộc của đề tài
Dự án yêu cầu nhóm 4 sinh viên tự tay cài đặt các thành phần cốt lõi của thuật toán **PPO (Proximal Policy Optimization)** kiểu **Actor-Critic** bằng PyTorch, sau đó huấn luyện tác tử thông minh điều khiển Mario chơi game NES Super Mario Bros trực tiếp từ điểm ảnh (pixel-to-action).

Theo đề bài chính thức (`Bài tập nhóm cuối kì.pdf`), nhóm phải hoàn thành 4 mục tiêu:
1. Chuyển hóa môi trường `mo-supermario-v0` thành bài toán RL chuẩn mực (quan sát tensor $4 \times 84 \times 84$, hành động rời rạc, phần thưởng vô hướng hóa, phân định rõ `terminated` và `truncated`).
2. Tự cài đặt đúng GAE, mục tiêu cắt PPO, mất mát Critic, entropy bonus và vòng cập nhật nhiều epoch qua mini-batch.
3. Huấn luyện, theo dõi và chẩn đoán đường dài (lợi tức episode, tọa độ $x$, tỷ lệ chạm cờ, approxKL, clipfrac, entropy).
4. Đánh giá tác tử theo giao thức cố định (20 lượt, seed 2026, xuất phát trễ $0..30$ NOOP), giải thích bằng số liệu định lượng nguyên nhân thành công hoặc thất bại.

### 1.2. Môi trường `mo-supermario-v0` và Hai lỗi đã được vá sẵn
- **Không gian quan sát:** Khung hình RGB $240 \times 256 \times 3$ $\to$ Tiền xử lý thành 4 ảnh xám $84 \times 84$ xếp chồng.
- **Không gian hành động:** Phím bấm NES $\to$ Tập `simple` (7 hành động kết hợp).
- **Phần thưởng véc-tơ 5 thành phần:** $[x\_pos, time, death, coin, enemy]$ $\to$ Chuyển thành số thực qua tích vô hướng $w \cdot r$ với bộ trọng số mặc định `(1.0, 1.0, 2.0, 0.05, 0.05)`.
- **Hai lỗi đã sửa sẵn trong kho mã:**
  1. `DeathPenaltyFix`: Khôi phục phần phạt $-25$ điểm khi Mario chết mà không chạm cờ.
  2. `XPosWrapFix`: Loại bỏ bước nhảy tràn số 16-bit của tọa độ $x$ khi qua các màn mê cung.

### 1.3. Thang điểm chính thức và Thứ tự ưu tiên (P0 / P1 / P2)

| Hạng mục | Điểm | Công thức & Tiêu chí nghiệm thu | Ưu tiên |
| :--- | :---: | :--- | :---: |
| **A. Cài đặt thuật toán PPO** | **30** | Tỷ lệ test công khai qua pytest (wrappers 6đ, model 6đ, GAE 9đ, PPO 9đ). Mốc: PASS 100%. | **P0** |
| **B. Vượt Màn 1-1** | **25** | $25 \cdot \min(1, \text{flag\_rate} / 0.8)$ theo đánh giá 20 lượt seed 2026. Đạt tối đa khi $\ge 16/20$ lượt thắng. | **P0** |
| **C. Vượt Màn thưởng** | **15** | $15 \cdot \min(1, \text{flag\_rate} / 0.8)$ trên màn tự chọn (Nhóm chọn Màn 1-2). | **P0** |
| **D. Báo cáo học thuật** | **25** | Tối đa 8 trang PDF, đầy đủ 6 mục quy chuẩn và 7 biểu đồ phân tích. | **P1** |
| **E. Trình bày & Demo** | **5** | 10 phút thuyết trình slide + 5 phút hỏi đáp vấn đáp trực tiếp. | **P1** |
| **Điểm thưởng mở rộng** | *+10* | Vượt thêm một màn ngoài danh sách (ví dụ 4-2, 7-1, 8-1). | **P2** |

> **Nguyên tắc phân bổ công sức:** Tuyệt đối không đánh đổi các nhiệm vụ P0 (chiếm 70 điểm) để làm P2 (+10 điểm). Chỉ khi Màn 1-1 và Màn thưởng đã đạt $\ge 0.8$ mới xem xét làm thêm.

### 1.4. Đối chiếu Kế hoạch cũ (5 tuần) và Kế hoạch mới (14 ngày)
- **Thuật toán:** Loại bỏ hoàn toàn DQN/Double DQN (không có trong đề bài); tập trung 100% tài nguyên CPU cho PPO Actor-Critic.
- **Môi trường:** Chuyển từ generic gym sang `mo-supermario-v0` với cơ chế `ScalarizeReward` vector 5 chiều.
- **Thời lượng:** Nén từ 35 ngày dàn trải thành Sprint tác chiến 14 ngày cuốn chiếu.
- **Phân công:** Phân định rõ 4 module code tương ứng các file `wrappers.py`, `model.py`, `ppo.py`, `train.py`/`evaluate.py`.
- **Màn chơi:** Bổ sung bắt buộc Màn thưởng 1-2 (Transfer Learning từ 1-1) và đánh giá chuẩn seed 2026.

---

## 2. KIẾN TRÚC KỸ THUẬT VÀ LUỒNG DỮ LIỆU TOÀN HỆ THỐNG

### 2.1. Sơ đồ luồng suy luận và luồng huấn luyện
1. **Luồng suy luận (Inference Flow):**
   $$\text{Env} \to \text{DeathPenaltyFix} \to \text{XPosWrapFix} \to \text{ButtonActions} \to \text{FrameSkip}(4) \to [\text{Tricks}] \to [\text{RandomNoop}] \to \text{ScalarizeReward} \to \text{GrayResize}(84) \to \text{FrameStack}(4) \to \text{ActorCritic} \to \text{Action}$$
2. **Luồng huấn luyện (Training Loop):**
   $$\text{Parallel Rollout }(T=256, N=16) \to \text{compute\_gae} \to \text{ppo\_update } (4\text{ epochs}, 8\text{ minibatches}) \to \text{Logging } (progress.csv) \to \text{Checkpoint } (500k\text{ steps}) \to \text{Eval}$$

### 2.2. Năm quy định kỹ thuật bất di bất dịch (Mục 3.5 đề bài)
1. **Tuyệt đối không sửa `mario_ppo/evaluate.py`.**
2. **Tuyệt đối không sửa khối `eval_mode` của `mario_ppo/env.py`.** Mẹo huấn luyện chỉ đặt trong `if not eval_mode:`.
3. **Đầu vào của mạng chỉ gồm điểm ảnh đã tiền xử lý.** Nghiêm cấm đọc RAM, không lấy $x\_pos$ hoặc `info` làm input.
4. **Không dùng checkpoint từ bên ngoài nhóm.** Toàn bộ trọng số phải tự huấn luyện.
5. **Mọi checkpoint phải tái tạo được từ file cấu hình đã nộp.**

---

## 3. PHÂN CHIA VAI TRÒ 4 THÀNH VIÊN & MA TRẬN RACI

| Thành viên | Tệp phụ trách chính | Trách nhiệm cốt lõi | Bài kiểm tra |
| :--- | :--- | :--- | :--- |
| **NGƯỜI 1** | `mario_ppo/wrappers.py` | `FrameSkip`, `GrayResize`, `ScalarizeReward`, mẹo `NoProgressTruncation`. | `tests/test_wrappers.py` (7/7 PASS) |
| **NGƯỜI 2** | `mario_ppo/model.py` | `ActorCritic.__init__` (CNN Nature, khởi tạo trực giao), `forward`, `act`. | `tests/test_model.py` (5/5 PASS) |
| **NGƯỜI 3** | `mario_ppo/ppo.py` | `compute_gae`, `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`, `ppo_loss`, `ppo_update`. | `tests/test_gae.py` (8/8) & `test_ppo.py` (10/10) |
| **NGƯỜI 4** | `train.py`, `evaluate.py`, `env.py` | Tích hợp pipeline, chạy full pytest, đo SPS, quản trị checkpoint/resume, benchmark seed 2026, tổng biên tập báo cáo & slide. | `tests/test_pipeline.py` & toàn bộ `pytest -q` (35/35 bài công khai) |

### Ma trận RACI tổng thể
- Người 1: **R/A** Wrappers; **C** Tích hợp; **R** Mục 1 & 5 Báo cáo; **R** Slide 1–3.
- Người 2: **R/A** Model; **C** Tích hợp; **R** Mục 1 & 2 Báo cáo; **R** Slide 4–5.
- Người 3: **R/A** PPO & GAE; **R/A** Ablation study; **R** Mục 1 & 4 Báo cáo; **R** Slide 6–7.
- Người 4: **R/A** Tích hợp & Train; **R/A** Benchmark; **A** Báo cáo 8 trang; **A** Slide & Đóng gói.

---

## 4. LỘ TRÌNH TÁC CHIẾN CHI TIẾT 14 NGÀY (DAY-BY-DAY TIMELINE)

### Giai đoạn 1: Ngày 1–3 – Thiết lập và Triển khai song song (P0)
- **Ngày 1:** Họp nhóm kick-off; thiết lập môi trường Python 3.10–3.12; phân nhánh Git; chạy `pytest -q` kiểm tra lỗi TODO ban đầu.
- **Ngày 2:** Triển khai đồng thời: GrayResize (Người 1); CNN Backbone (Người 2); compute_gae (Người 3); script huấn luyện (Người 4).
- **Ngày 3:** Hoàn thiện độc lập: FrameSkip, ScalarizeReward (Người 1 qua `test_wrappers`); forward, act (Người 2 qua `test_model`); policy/value loss (Người 3 qua `test_gae`).

### Giai đoạn 2: Ngày 4–5 – Tích hợp và Kiểm thử PPO (P0 Milestone)
- **Ngày 4:** Người 3 hoàn thiện `ppo_loss`, `ppo_update` (qua `test_ppo`). Người 4 merge nhánh integration, sửa lỗi tương thích tensor giữa 3 module.
- **Ngày 5 (MỐC TUẦN 1 CỦA ĐỀ BÀI):** Toàn bộ `pytest -q` PASS 100% (35/35 bài test công khai, cộng các bài bổ sung `tests/test_nhom_*.py`). Chạy smoke test huấn luyện 10k bước; đo SPS ban đầu.

### Giai đoạn 3: Ngày 6–9 – Huấn luyện và Đánh giá Màn 1-1 (P0)
- **Ngày 6:** Khởi chạy Run 1 Màn 1-1 (5M bước) trên máy trạm đa lõi mạnh nhất. Đo SPS thời gian thực.
- **Ngày 7:** Đọc nhật ký `progress.csv` sau 1.5–2M bước. Thêm mẹo `NoProgressTruncation` nếu Mario kẹt chân ống nước; chạy Run 2 song song.
- **Ngày 8:** Cán mốc 3.5–5M bước. Lọc các checkpoint có `flag_rate` cao trong `evals.csv`. Xuất biểu đồ đường cong học sơ bộ.
- **Ngày 9 (CỘT MỐC ĐÁNH GIÁ 1-1):** Đánh giá chuẩn Màn 1-1 qua `evaluate.py` (seed 2026, 20 lượt) đạt `flag_rate >= 0.8` (25 điểm). Khởi chạy Transfer Learning cho Màn thưởng 1-2.

### Giai đoạn 4: Ngày 9–12 – Màn thưởng và Thí nghiệm so sánh đối chứng (P0 & P1)
- **Ngày 10:** Huấn luyện Màn 1-2 từ checkpoint 1-1; Người 3 & Người 1 chạy Thí nghiệm đối chứng $\gamma=0.9$ vs $\gamma=0.99$ trên Màn 1-1 (2M bước, seed=1).
- **Ngày 11:** Màn thưởng cán mốc huấn luyện; thu thập số liệu Thí nghiệm đối chứng; trích xuất tọa độ chết làm Phân tích thất bại.
- **Ngày 12 (CỘT MỐC ĐÁNH GIÁ MÀN THƯỞNG):** Đánh giá chuẩn Màn 1-2 đạt `flag_rate >= 0.8` (15 điểm). Xuất 7 biểu đồ phân tích chuẩn 300 DPI.

### Giai đoạn 5: Ngày 13–14 – Hoàn thiện sản phẩm, Báo cáo, Slide và Vấn đáp (P1)
- **Ngày 13:** Đóng băng mã nguồn (tag `v1.0-final`). 4 thành viên viết Báo cáo học thuật 6 mục; khóa file PDF đúng tối đa 8 trang; làm Slide 10 phút.
- **Ngày 14:** Tổ chức 2 buổi diễn tập thuyết trình và phản biện vấn đáp nội bộ; kiểm tra tính tái lập trên máy mới; đóng gói tệp `nhom-<so>.zip` nộp bài.

---

## 5. CHIẾN LƯỢC HUẤN LUYỆN, TÀI NGUYÊN TÍNH TOÁN VÀ ĐÁNH GIÁ ĐỊNH LƯỢNG

### 5.1. Ma trận thời gian huấn luyện theo SPS
Trình giả lập NES chạy trên CPU; mạng CNN nhỏ nên GPU không tăng tốc nhiều. Tốc độ phụ thuộc số nhân CPU:
- **100 SPS (Laptop 4 nhân):** 3.5M bước mất ~9.7 giờ; 5M bước mất ~13.9 giờ.
- **200 SPS (Laptop Gaming 6-8 nhân):** 3.5M bước mất ~4.9 giờ; 5M bước mất ~6.9 giờ.
- **350 SPS (Desktop 8-12 nhân, 16 envs):** 3.5M bước mất ~2.8 giờ; 5M bước mất ~4.0 giờ.
- **500 SPS (Máy trạm 16 nhân, 16 envs):** 3.5M bước mất ~1.9 giờ; 5M bước mất ~2.8 giờ.

### 5.2. Quản lý Checkpoints, Resume và Nhận diện sớm lượt chạy hỏng
- Lưu checkpoint mỗi 500k bước kèm 5 lượt đánh giá định kỳ (`evals.csv`) và video lượt tốt nhất (`videos/step_*.mp4`).
- Khi tiến trình bị dừng, chạy tiếp bằng: `python -m mario_ppo.train resume=<checkpoint.pt>`.
- **Dấu hiệu cảnh báo đỏ (Dừng ngay để tinh chỉnh):**
  - $x\_pos < 400$ kéo dài sau 1M bước: Mario kẹt ống nước $\to$ Bật `NoProgressTruncation(max_steps=200)`.
  - Entropy giảm dốc $< 0.1$ sau 300k bước: Sụp đổ khám phá $\to$ Tăng `ent_coef` lên 0.015 hoặc 0.02.
  - ApproxKL $> 0.05$ liên tục: Bước nhảy quá lớn $\to$ Kích hoạt `target_kl=0.02`, giảm learning rate.

### 5.3. Chiến lược Màn thưởng: Học chuyển giao (Transfer Learning) Màn 1-2
- Chọn **Màn 1-2** vì địa hình hầm ngầm bằng phẳng, trần đá thấp, không có hố sâu tử thần, ít rủi ro rơi tự do.
- Khởi tạo trọng số từ Checkpoint tốt nhất của Màn 1-1 giúp tiết kiệm 60% thời gian (chỉ cần 3–5M bước là hội tụ).

### 5.4. Thí nghiệm so sánh đối chứng (Ablation Study)
- **Yếu tố thay đổi duy nhất:** Hệ số chiết khấu $\gamma = 0.9$ (Baseline) so với $\gamma = 0.99$ (Ablation).
- **Giả thuyết khoa học:** Mario là game dài hơi, $\gamma = 0.99$ có vẻ nhìn xa hơn nhưng lại làm tăng phương sai của Critic, khiến gradient dao động mạnh và học chậm hơn ở giai đoạn đầu so với $\gamma = 0.9$.
- **Cấu hình đối chứng:** Cùng Màn 1-1, cùng seed=1, cùng 2,000,000 bước, cùng 16 envs.

### 5.5. Giao thức Đánh giá Chuẩn mực
```powershell
python -m mario_ppo.evaluate --ckpt <checkpoint.pt> --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4
```
Chạy 20 lượt độc lập, mỗi lượt xuất phát sau $0..30$ NOOP sinh từ seed 2026, chế độ `eval_mode=True`. Đạt khi `flag_rate >= 0.8` (16/20 lượt).

---

## 6. QUY CHUẨN BÁO CÁO HỌC THUẬT 8 TRANG, SLIDE THUYẾT TRÌNH VÀ NGÂN HÀNG VẤN ĐÁP

### 6.1. Báo cáo học thuật (Tối đa đúng 8 trang PDF)
1. **Mục 1 - Cài đặt (1.5 trang):** Mô tả các thành phần đã viết, bảng kết quả pytest 100% PASS.
2. **Mục 2 - Thiết lập huấn luyện (1.0 trang):** Bảng siêu tham số, cấu hình phần cứng, số bước, SPS.
3. **Mục 3 - Đường cong học (2.0 trang):** 6 biểu đồ (return, x_pos, flag_rate, entropy, approxKL, clipfrac) kèm nhận xét cơ chế.
4. **Mục 4 - Thí nghiệm so sánh (1.5 trang):** Giả thuyết, thiết lập đối chứng $\gamma=0.9$ vs $0.99$, đồ thị so sánh và kết luận.
5. **Mục 5 - Phân tích thất bại (1.0 trang):** Biểu đồ phân bố điểm chết (Failure Histogram) và giải thích nguyên nhân.
6. **Mục 6 - Phân công (1.0 trang):** Bảng đóng góp phần trăm và ma trận phân công 4 thành viên.

### 6.2. Kịch bản Thuyết trình Slide (10 phút)
- **Phút 00:00 – 02:30 (Người 1):** Bối cảnh bài toán, môi trường `mo-supermario-v0`, chuỗi Wrapper.
- **Phút 02:30 – 05:00 (Người 2):** Kiến trúc Actor-Critic CNN, khởi tạo trực giao gain 0.01.
- **Phút 05:00 – 07:30 (Người 3):** Lý thuyết GAE, PPO clipped objective, approxKL, ppo_update.
- **Phút 07:30 – 10:00 (Người 4):** Kết quả Màn 1-1 và Màn thưởng, Ablation study, video Demo vượt màn.

### 6.3. Ngân hàng 26 Câu hỏi Vấn đáp Trọng tâm
*(Xem chi tiết toàn văn câu hỏi và câu trả lời mẫu trong file Word hoặc sổ tay từng người).*
- **Người 1:** Sprite flickering & max 2 frame; GrayResize Luminance vs RGB mean; FrameStack 4 khung; terminated vs truncated; ScalarizeReward.
- **Người 2:** Shared CNN trunk; Policy head gain 0.01 (entropy ban đầu $\approx 1.94$); Critic head gain 1.0; chuẩn hóa /255.0; `@torch.no_grad()`.
- **Người 3:** GAE Bias-Variance trade-off; Clipped Surrogate pessimistic bound; approxKL $k_3$ không âm; chuẩn hóa advantage; `unbiased=False`.
- **Người 4:** `AsyncVectorEnv` forkserver chống deadlock; `SAME_STEP` & `final_obs`; kích thước lô $T=256, N=16$; Anneal lr; seed 2026 trong evaluate.
- **Toàn nhóm:** Tại sao PPO hơn DQN trong Mario; tại sao cấm đưa tọa độ $x$ vào mạng; vì sao tiêu chí là flag_rate $\ge 0.8$; kế hoạch dự phòng khi Màn 1-1 chạy chậm.

---

## 7. QUẢN TRỊ MÃ NGUỒN, TÍNH TÁI LẬP VÀ ĐÓNG GÓI HỒ SƠ NỘP BÀI

### 7.1. Cấu trúc tệp nén `nhom-<so>.zip` (Mục 7 đề bài)
1. Toàn bộ mã nguồn: `mario_ppo/`, `configs/`, `tests/`, `pyproject.toml`.
2. Cấu hình đã dùng: `runs/1-1/.../config.yaml`, `runs/1-2/.../config.yaml`.
3. Nhật ký huấn luyện: `progress.csv` và `evals.csv` của các lần chạy nộp.
4. Checkpoints tốt nhất: `step_*.pt` cho Màn 1-1 và Màn thưởng.
5. Tệp JSON đánh giá: `ket-qua-1-1.json` và `ket-qua-1-2.json`.
6. Video MP4 minh chứng: `1-1.mp4` và `1-2.mp4`.
7. Báo cáo học thuật: `Bao_Cao_Nhom_<so>.pdf` (Độ dài $\le 8$ trang).
8. Slide thuyết trình: `Slide_Thuyet_Trinh_Nhom_<so>.pptx / .pdf`.

---

## 8. BẢNG QUẢN TRỊ 15 RỦI RO KỸ THUẬT VÀ KẾ HOẠCH ỨNG PHÓ

*(Tóm lược 6 rủi ro cốt tử nhất - xem đầy đủ 15 rủi ro trong tài liệu Word):*
1. **Module không qua pytest:** Bắt buộc hoàn thành trước Ngày 4; Người 4 phối hợp debug trực tiếp.
2. **Actor-Critic lệch tensor shape với PPO:** Quy chuẩn: `value` shape `(B,)` float32, `logits` shape `(B, n_actions)`.
3. **GAE xử lý sai bootstrap:** Tuân thủ: `terminated` không bootstrap; `truncated` bootstrap từ `final_values`.
4. **Pipeline deadlock đa tiến trình:** Dùng `context="forkserver"` trong `AsyncVectorEnv`.
5. **Mario kẹt ở ống nước đầu tiên:** Kích hoạt `NoProgressTruncation(max_steps=200)` trong `env.py`.
6. **Báo cáo vượt quá 8 trang:** Người 4 biên tập lại hình ảnh, gộp bảng, tinh chỉnh khoảng cách dòng 1.15 để khóa đúng 8 trang.

---

## 9. BỘ TIÊU CHÍ NGHIỆM THU VÀ CHECKLIST TOÀN DIỆN

- [ ] 1. Toàn bộ 35 bài test công khai `pytest -q` PASS 100% (30đ Hạng mục A).
- [ ] 2. Màn 1-1 đạt `flag_rate >= 0.8` (25đ Hạng mục B). Có `ket-qua-1-1.json` và `1-1.mp4`.
- [ ] 3. Màn thưởng 1-2 đạt `flag_rate >= 0.8` (15đ Hạng mục C). Có `ket-qua-1-2.json` và `1-2.mp4`.
- [ ] 4. Hoàn thành Thí nghiệm đối chứng có kiểm soát $\gamma=0.9$ vs $\gamma=0.99$.
- [ ] 5. Báo cáo học thuật hoàn chỉnh, đủ 6 mục, 7 biểu đồ, độ dài chính xác $\le 8$ trang PDF (25đ Hạng mục D).
- [ ] 6. Slide thuyết trình 10 phút sẵn sàng, video demo chuẩn nét, cả nhóm tự tin trả lời vấn đáp (5đ Hạng mục E).
- [ ] 7. Đóng gói đầy đủ hồ sơ `nhom-<so>.zip` và kiểm tra tính tái lập trên máy độc lập thành công.
