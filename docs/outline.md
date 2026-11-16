# DÀN Ý CHI TIẾT BÁO CÁO ĐỒ ÁN

> **Trạng thái:** Bản dàn ý v1, chờ cả nhóm thông qua tại họp Sprint 1 (Tuần 1, Ngày 7).
> **Người lập:** Người 4 (WBS #6: "Lập dàn ý chi tiết Báo cáo và phân công các mục lý thuyết").
> **Căn cứ:** `Phan_Cong_Nhiem_Vu/KE_HOACH_TONG_THE_5_TUAN.md` (mục 4, 5, 7, 10, 12) và sổ tay Người 1–4.
> Các bản thảo nộp dưới dạng `docs/chapterN.md` (theo sổ tay từng người); Người 4 hợp nhất thành Word/PDF ở Tuần 5.

Ký hiệu dùng trong file: **N1–N4** = Người 1–4; **T** = Tuần; **N(ngày)** = Ngày thứ mấy của dự án (vd *T2-N14* = Tuần 2, Ngày 14).
Hình và bảng đánh số theo chương (Hình 3.1, Bảng 7.1...). Tên file hình gợi ý: `docs/figures/chN_*.png|svg`.

---

## 0. THÔNG TIN CHUNG

| Mục | Nội dung |
| :--- | :--- |
| Tên đề tài | Giải trò chơi Super Mario Bros (World 1-1) bằng Học tăng cường sâu: Dueling Double DQN và đối sánh với PPO |
| Môn học | Học tăng cường (Reinforcement Learning) |
| Môi trường | `gym-super-mario-bros` + `nes-py` trên Gymnasium (phiên bản ghim trong `requirements.txt`, Python 3.13 64-bit theo `docs/ENVIRONMENT.md`) |
| Nhóm | 04 thành viên (N1: Môi trường & Tiền xử lý; N2: Thuật toán; N3: Huấn luyện, Reward & Thực nghiệm; N4: Tích hợp, Kiểm thử & Điều phối học thuật) |
| Mục tiêu | Phần 1 (8.0đ): tác nhân chạm cờ World 1-1, Win Rate ≥ 80% trên 20 episode đánh giá. Phần 2 (2.0đ): đối sánh DDQN vs PPO, ablation reward và không gian hành động |
| Sản phẩm | `Bao_cao_RL_Mario.pdf` + bản DOCX (Tuần 5, Ngày 30–32) |

### Phần mở đầu (N4 viết chính, cả nhóm rà soát, hạn T5-N31)

| Thành phần | Nội dung chính | Ghi chú |
| :--- | :--- | :--- |
| Trang bìa (bìa cứng + bìa lót) | Trường/khoa, tên môn, tên đề tài, GVHD, danh sách 4 thành viên (họ tên, MSSV, vai trò), nơi và thời gian | Không đánh số trang |
| Lời cam đoan | Cam kết kết quả do nhóm thực hiện, số liệu trung thực, trích dẫn đầy đủ; chữ ký 4 thành viên | |
| Lời cảm ơn | Cảm ơn GVHD, khoa, tác giả mã nguồn mở được sử dụng | |
| Tóm tắt (Tiếng Việt, có thể kèm Abstract tiếng Anh) | Bài toán, phương pháp (pipeline tiền xử lý, Dueling DDQN, reward shaping), kết quả chính (Win Rate, Max X, so sánh PPO), đóng góp; 200–300 từ + 4–6 từ khóa | Viết sau cùng, khi đã có số liệu Ch7–8 |
| Mục lục | Tự động từ Heading 1–3 | |
| Danh mục hình vẽ | Tự động từ caption "Hình" | |
| Danh mục bảng biểu | Tự động từ caption "Bảng" | |
| Danh mục từ viết tắt | RL, DRL, MDP, POMDP, DQN, DDQN, PPO, GAE, CNN, NES, FPS, HUD, TD, MSE, ε-greedy, SB3, ... (sắp xếp A–Z, kèm nghĩa tiếng Anh và tiếng Việt) | Mỗi người gửi thuật ngữ trong chương mình cùng bản thảo |

Đánh số trang phần mở đầu bằng số La Mã (i, ii, ...); từ Chương 1 dùng số Ả Rập.

---

## 1. NỘI DUNG 12 CHƯƠNG

### CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI VÀ MỤC TIÊU NGHIÊN CỨU
Viết chính: **N1** · Rà soát: N4 · Hạn bản thảo: **T2-N14**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 1.1. Bối cảnh và lý do chọn đề tài | Vị trí của trò chơi điện tử làm benchmark cho DRL; Super Mario Bros (NES, 1985) khó hơn Atari cổ điển | Hình 1.1: Ảnh chụp màn hình World 1-1 | N1 | T2-N14 |
| 1.2. Thách thức của bài toán Mario | Cuộn ngang, kẻ thù di động, phần thưởng trễ/thưa, chuỗi hành động phụ thuộc thời gian (chạy lấy đà rồi nhảy) | Bảng 1.1: So sánh Mario với Pong/Breakout | N1 | T2-N14 |
| 1.3. Mục tiêu nghiên cứu | | | N1 | T2-N14 |
| 1.3.1. Mục tiêu chính (8 điểm) | Chạm cờ World 1-1 (`flag_get == True`), Win Rate ≥ 80%, pipeline hoàn chỉnh | | N1 | T2-N14 |
| 1.3.2. Mục tiêu mở rộng (2 điểm) | DDQN vs PPO; ablation reward; `RIGHT_ONLY` vs `SIMPLE_MOVEMENT`; TensorBoard và video demo | | N1 | T2-N14 |
| 1.4. Phạm vi và giới hạn | Chỉ World 1-1, một môi trường đơn, tài nguyên GPU cá nhân/Colab | | N1 | T2-N14 |
| 1.5. Phương pháp tiếp cận tổng quát | Tóm lược: tiền xử lý ảnh → Dueling DDQN → reward shaping → đánh giá độc lập → đối sánh | Hình 1.2: Sơ đồ phương pháp tổng quát | N1 | T2-N14 |
| 1.6. Cấu trúc báo cáo | Tóm tắt một câu cho mỗi chương 2–12 | | N1 | T2-N14 |

### CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VỀ HỌC TĂNG CƯỜNG
Viết chính: **N2** · Rà soát: N3 · Hạn bản thảo: **T2-N14**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 2.1. Tổng quan học tăng cường | Tác nhân, môi trường, trạng thái, hành động, phần thưởng, chính sách; so với học có/không giám sát | Hình 2.1: Vòng lặp tác nhân–môi trường | N2 | T2-N14 |
| 2.2. Quá trình quyết định Markov | | | N2 | T2-N14 |
| 2.2.1. Định nghĩa MDP (S, A, P, R, γ) | Tính chất Markov, lợi nhuận chiết khấu G_t | | N2 | T2-N14 |
| 2.2.2. Hàm giá trị và phương trình Bellman | V^π, Q^π, phương trình kỳ vọng và tối ưu Bellman | | N2 | T2-N14 |
| 2.2.3. MDP quan sát một phần (POMDP) | Khái niệm để làm nền cho frame stacking ở Ch3 | | N2 | T2-N14 |
| 2.3. Học giá trị dựa trên sai phân thời gian | TD learning, Q-Learning [8], khám phá/khai thác, ε-greedy | | N2 | T2-N14 |
| 2.4. Deep Q-Network (DQN) | Xấp xỉ hàm bằng CNN, Experience Replay [9], Target Network [1]; Huber loss [23] | Hình 2.2: Sơ đồ DQN với replay buffer và target network | N2 | T2-N14 |
| 2.5. Double DQN | Overestimation bias của toán tử max [7]; tách chọn và đánh giá hành động [2] | Bảng 2.1: Công thức target DQN vs Double DQN | N2 | T2-N14 |
| 2.6. Dueling Network | Tách luồng V(s) và A(s,a), phép trừ trung bình để đảm bảo tính định danh [3] | Hình 2.3: Kiến trúc Dueling (theo [3]) | N2 | T2-N14 |
| 2.7. Phương pháp Policy Gradient và PPO | Policy gradient, Actor-Critic, GAE [11], Clipped Surrogate Objective [4]; on-policy vs off-policy | Bảng 2.2: So sánh DDQN và PPO | N2 | T2-N14 |
| 2.8. Các công trình liên quan | DQN trên Atari/ALE [1][6][12], Rainbow [14], PER [15]; các dự án Mario dùng gym-super-mario-bros | | N2 | T2-N14 |

### CHƯƠNG 3. PHÂN TÍCH BÀI TOÁN SUPER MARIO BROS DƯỚI GÓC NHÌN HỌC TĂNG CƯỜNG
Viết chính: **N1** · Rà soát: N2 · Hạn bản thảo: **T2-N14**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 3.1. Môi trường giả lập | gym-super-mario-bros, nes-py, chuẩn Gymnasium [16][17][18]; `reset()` → `(obs, info)`, `step()` → 5 giá trị | Bảng 3.1: Phiên bản thư viện (theo `requirements.txt`) | N1 | T2-N14 |
| 3.2. Mô hình hóa MDP cho World 1-1 | | | N1 | T2-N14 |
| 3.2.1. Không gian trạng thái | Khung hình RGB 240×256×3; các trường `info` (`x_pos`, `flag_get`, ...) | | N1 | T2-N14 |
| 3.2.2. Không gian hành động | 256 tổ hợp phím → `SIMPLE_MOVEMENT` (7), `RIGHT_ONLY` (5), `RIGHT_JUMP` (2) qua `JoypadSpace` | Bảng 3.2: Danh sách 7 hành động `SIMPLE_MOVEMENT` | N1 | T2-N14 |
| 3.2.3. Phần thưởng gốc và điều kiện kết thúc | Reward native, `terminated` (thắng/chết) vs `truncated` (TimeLimit 4500 bước quyết định) | | N1 | T2-N14 |
| 3.3. Pipeline tiền xử lý quan sát | | Hình 3.1: Sơ đồ pipeline wrapper (theo `docs/ENVIRONMENT.md`) | N1 | T2-N14 |
| 3.3.1. Frame skip 4 và max-pooling 2 frame cuối | Giảm chi phí tính toán; xử lý sprite flickering của NES | Hình 3.2: Ví dụ flicker trước/sau max-pool | N1 | T2-N14 |
| 3.3.2. Grayscale và resize 84×84 | Giảm chiều dữ liệu (Pillow, nội suy BOX) | Hình 3.3: Khung hình gốc → xám → 84×84 | N1 | T2-N14 |
| 3.3.3. Frame stack 4 | Suy ra vận tốc/hướng chuyển động; quan sát `(4, 84, 84)` uint8; giới hạn: không bảo đảm quan sát đầy đủ | Hình 3.4: Minh họa 4 frame xếp chồng | N1 | T2-N14 |
| 3.4. Baseline tác nhân ngẫu nhiên | Kết quả `baseline_random.py`: agent steps/s, emulator FPS, RSS, max X | Bảng 3.3: Kết quả baseline random | N1 | T2-N14 |

### CHƯƠNG 4. THIẾT KẾ HỆ THỐNG VÀ KIẾN TRÚC MÔ HÌNH
Viết chính: **N2** · Rà soát: N1 · Hạn bản thảo: **T3-N21**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 4.1. Kiến trúc tổng thể hệ thống | Các module env / models / agents / rewards / utils và luồng dữ liệu giữa chúng | Hình 4.1: Sơ đồ khối tổng thể; Bảng 4.1: Cấu trúc thư mục repo | N2 | T3-N21 |
| 4.2. Kiến trúc mạng Dueling CNN | | Hình 4.2: Kiến trúc mạng Dueling Convolutional Neural Network | N2 | T3-N21 |
| 4.2.1. Bộ trích đặc trưng Nature CNN | 3 lớp conv theo [1], kích thước tensor qua từng lớp | Bảng 4.2: Kích thước tensor và số tham số từng lớp | N2 | T3-N21 |
| 4.2.2. Hai luồng Value và Advantage | Công thức hợp nhất Q = V + (A − mean A) | | N2 | T3-N21 |
| 4.3. Replay Buffer | Dung lượng 100.000 transition, lưu uint8, lấy mẫu đồng nhất, ước tính bộ nhớ | Hình 4.3: Luồng ghi/đọc replay buffer | N2 | T3-N21 |
| 4.4. Tác nhân Dueling Double DQN | | | N2 | T3-N21 |
| 4.4.1. Chọn hành động ε-greedy | Lịch giảm ε tuyến tính | | N2 | T3-N21 |
| 4.4.2. Tính target và hàm mất mát | Double Q target, Huber loss, gradient clipping (max_norm = 10) | | N2 | T3-N21 |
| 4.4.3. Cập nhật mạng đích | Hard update mỗi 10.000 bước | | N2 | T3-N21 |
| 4.5. Vòng lặp huấn luyện và checkpoint | Mã giả `train.py`; `save_checkpoint`/`load_checkpoint` | Hình 4.4: Lưu đồ vòng lặp huấn luyện; Thuật toán 4.1 (mã giả) | N2 | T3-N21 |
| 4.6. Kiểm thử tương thích tensor | Input `(B, 4, 84, 84)` → output `(B, 7)`; kết quả pytest | Bảng 4.3: Danh sách unit test mô hình | N2 | T3-N21 |

### CHƯƠNG 5. KỸ THUẬT THIẾT KẾ HÀM PHẦN THƯỞNG (REWARD SHAPING)
Viết chính: **N3** · Rà soát: N2 · Hạn bản thảo: **T3-N21**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 5.1. Cơ sở lý thuyết reward shaping | Potential-based shaping, tính bất biến chính sách [10]; Φ(s) = x_pos | | N3 | T3-N21 |
| 5.2. Phân tích các bẫy hành vi | | Hình 5.1: Bản đồ World 1-1 đánh dấu vị trí bẫy (ống nước, hố) | N3 | T3-N21 |
| 5.2.1. Bẫy sợ chết / chạy lùi | Biểu hiện, nguyên nhân, dấu hiệu trên log | | N3 | T3-N21 |
| 5.2.2. Bẫy kẹt ống nước cao | X ≈ 400–500 | | N3 | T3-N21 |
| 5.2.3. Bẫy rơi hố | Các hố trên đường tới cờ | | N3 | T3-N21 |
| 5.2.4. Bẫy nhảy tại chỗ câu giờ | Chỉ thưởng khi vượt X_max | | N3 | T3-N21 |
| 5.3. Reward v1 | Tiến độ X (clip), phạt thời gian, phạt chết | Bảng 5.1: Thành phần và trọng số Reward v1 | N3 | T3-N21 |
| 5.4. Reward v2 | Bổ sung anti-stuck, thưởng cờ +500, khuyến khích nhảy vượt chướng ngại | Bảng 5.2: So sánh thành phần Reward v1 vs v2 | N3 | T3-N21 |
| 5.5. Cài đặt `MarioCustomRewardWrapper` | Vị trí trong pipeline (reward hook của N1), tham số cấu hình | Hình 5.2: Vị trí reward wrapper trong pipeline | N3 | T3-N21 |

### CHƯƠNG 6. THIẾT LẬP HUẤN LUYỆN VÀ QUY TRÌNH THỰC NGHIỆM
Viết chính: **N3** · Rà soát: N4 · Hạn bản thảo: **T3-N21**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 6.1. Môi trường phần cứng và phần mềm | CPU/GPU, RAM, OS, Colab/Kaggle nếu dùng, phiên bản thư viện | Bảng 6.1: Cấu hình phần cứng/phần mềm | N3 | T3-N21 |
| 6.2. Siêu tham số | lr 1e-4 (Adam [24]), batch 64, γ 0.99, ε 1.0→0.05 trong 1M bước, buffer 100k, target update 10k, tổng 1–3M bước | Bảng 6.2: Bảng siêu tham số Dueling DDQN | N3 | T3-N21 |
| 6.3. Giám sát bằng TensorBoard | Các chỉ số Episode_Reward, Max_X_Position, Episode_Length, Huber_Loss, Epsilon; dấu hiệu tốt/xấu | Bảng 6.3: Ý nghĩa các chỉ số TensorBoard | N3 | T3-N21 |
| 6.4. Quy trình huấn luyện thực tế | Các đợt chạy, mốc 500k/1M/2M bước, sự cố và cách xử lý, chuyển Reward v1 → v2 | Hình 6.1: Dòng thời gian các đợt huấn luyện | N3 | T3-N21 |
| 6.5. Quản lý checkpoint và tái lập | Lưu định kỳ, top-k checkpoint, cố định seed (random/numpy/torch), sao lưu | | N3 | T3-N21 |
| 6.6. Giao thức đánh giá | ε = 0, 20 episode, chỉ số Win Rate / Avg Max X / Avg Steps; lưu ý từ `docs/ENVIRONMENT.md`: World 1-1 khởi tạo tất định nên đổi seed không tạo biến thiên, cần giao thức tạo biến thiên thống nhất (vd no-op/sticky action theo [13]) | Bảng 6.4: Định nghĩa các chỉ số đánh giá | N3 (N4 cung cấp `evaluate.py`) | T3-N21 |

### CHƯƠNG 7. KẾT QUẢ THỰC NGHIỆM GIẢI MÀN WORLD 1-1
Viết chính: **N3** · Rà soát: N4 · Hạn bản thảo: **T4-N28**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 7.1. Đường cong học tập | Reward, Max X, Episode Length, Loss theo số bước | Hình 7.1–7.4: Các đồ thị TensorBoard (xuất vector) | N3 | T4-N28 |
| 7.2. Kết quả đánh giá độc lập | Kết quả `evaluate.py` trên checkpoint tốt nhất (`results/eval_results.csv`) | Bảng 7.1: Bảng kết quả đánh giá thực nghiệm trên 20 episode độc lập | N3 (số liệu: N4) | T4-N28 |
| 7.3. So sánh với baseline ngẫu nhiên | Random agent vs Dueling DDQN | Bảng 7.2: Random vs DDQN (Win Rate, Max X) | N3 | T4-N28 |
| 7.4. Phân tích hành vi | | | N3 | T4-N28 |
| 7.4.1. Các pha xử lý tiêu biểu | Vượt ống nước, nhảy qua hố, né/giẫm Goomba | Hình 7.5: Chuỗi khung hình pha nhảy qua ống nước | N3 | T4-N28 |
| 7.4.2. Phân tích Q-value và V(s) | Diễn biến Max Q trong một episode (từ HUD của `demo.py`) | Hình 7.6: Q-value theo thời gian trong một episode | N3 | T4-N28 |
| 7.4.3. Các trường hợp thất bại | Vị trí và nguyên nhân các episode không về đích | Bảng 7.3: Thống kê nguyên nhân thất bại | N3 | T4-N28 |
| 7.5. Video demo | Mô tả video MP4 kèm HUD trong `recordings/`, link dự phòng | Hình 7.7: Ảnh chụp video demo có HUD | N3 (video: N4) | T4-N28 |
| 7.6. Nghiệm thu mục tiêu 8 điểm | Đối chiếu tiêu chí nghiệm thu mục 10 của kế hoạch | Bảng 7.4: Bảng đối chiếu tiêu chí nghiệm thu | N3 | T4-N28 |

### CHƯƠNG 8. CÁC NGHIÊN CỨU MỞ RỘNG VÀ ABLATION STUDY
Viết chính: **N2, N3** · Rà soát: N4 · Hạn bản thảo: **T4-N28**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 8.1. Thiết kế thí nghiệm | Biến độc lập, biến cố định, số bước, số lần chạy/seed huấn luyện, chỉ số so sánh [21][22] | Bảng 8.1: Ma trận các thí nghiệm | N2, N3 | T4-N28 |
| 8.2. Đối sánh Dueling DDQN và PPO | | | N2 | T4-N28 |
| 8.2.1. Cài đặt PPO | Actor-Critic tự viết hoặc Stable-Baselines3 [20]; cùng pipeline quan sát | Bảng 8.2: Siêu tham số PPO | N2 | T4-N28 |
| 8.2.2. Kết quả và tốc độ hội tụ | Cùng ngân sách bước môi trường; sample efficiency, độ ổn định, thời gian huấn luyện | Hình 8.1: Đường cong DDQN vs PPO; Bảng 8.3: So sánh định lượng | N2 | T4-N28 |
| 8.3. Ablation hàm phần thưởng | Reward gốc/v1 vs v2 (500k bước mỗi biến thể) | Hình 8.2: Đường cong Max X theo reward; Bảng 8.4 | N3 | T4-N28 |
| 8.4. Ablation không gian hành động | `RIGHT_ONLY` (5) vs `SIMPLE_MOVEMENT` (7) (môi trường do N1 cung cấp) | Hình 8.3: Đường cong theo action space; Bảng 8.5 | N3 | T4-N28 |
| 8.5. Tổng hợp và nhận xét | Kết luận rút ra từ các thí nghiệm, độ tin cậy thống kê | Bảng 8.6: Tổng hợp kết quả các thí nghiệm | N2, N3 | T4-N28 |

### CHƯƠNG 9. HẠN CHẾ CỦA HỆ THỐNG VÀ HƯỚNG PHÁT TRIỂN
Viết chính: **N1** · Rà soát: N3 · Hạn bản thảo: **T4-N28**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 9.1. Hạn chế | | | N1 | T4-N28 |
| 9.1.1. Quá khớp World 1-1 | Tác nhân học thuộc màn chơi tất định | | N1 | T4-N28 |
| 9.1.2. Khả năng tổng quát hóa | Thử nghiệm checkpoint trên World 1-2 | Bảng 9.1: Kết quả thử trên World 1-2 | N1 | T4-N28 |
| 9.1.3. Tài nguyên và thời gian huấn luyện | RAM replay buffer, FPS, chi phí GPU | | N1 | T4-N28 |
| 9.1.4. Giới hạn của giao thức đánh giá | Tính tất định của môi trường, số lần chạy huấn luyện hạn chế | | N1 | T4-N28 |
| 9.2. Hướng phát triển | PER [15], Rainbow [14], n-step, vector hóa môi trường, curriculum nhiều màn, đầu vào RGB | Bảng 9.2: Đề xuất cải tiến và lợi ích dự kiến | N1 | T4-N28 |

### CHƯƠNG 10. HƯỚNG DẪN CÀI ĐẶT, SỬ DỤNG VÀ KIỂM THỬ TÁI LẬP
Viết chính: **N4** · Rà soát: N1 · Hạn bản thảo: **T4-N28** (bản hoàn chỉnh T5-N30)

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 10.1. Yêu cầu hệ thống | Python 3.13 64-bit, OS, GPU tùy chọn; Build Tools C++ nếu phải build nes-py trên Windows | Bảng 10.1: Yêu cầu phần cứng/phần mềm | N4 | T4-N28 |
| 10.2. Cài đặt | Clone repo, tạo venv, `pip install -r requirements.txt` (+ `requirements-dev.txt`) | | N4 | T4-N28 |
| 10.3. Sử dụng | | | N4 | T4-N28 |
| 10.3.1. Huấn luyện | Lệnh `train.py`, file cấu hình trong `configs/` | | N4 | T4-N28 |
| 10.3.2. Đánh giá | `evaluate.py --checkpoint ... --episodes 20` | | N4 | T4-N28 |
| 10.3.3. Demo và ghi video | `demo.py`, đầu ra `recordings/*.mp4` | | N4 | T4-N28 |
| 10.3.4. Theo dõi TensorBoard | `tensorboard --logdir logs/` | | N4 | T4-N28 |
| 10.4. Kiểm thử tự động | Unit/integration test với pytest (`tests/`), marker `integration` | Bảng 10.2: Danh sách test và kết quả | N4 | T4-N28 |
| 10.5. Kiểm thử tái lập trên máy sạch | Quy trình clone → cài → evaluate → demo trên máy thứ hai; không hard-code đường dẫn | Bảng 10.3: Kết quả kiểm thử tái lập | N4 | T4-N28 |
| 10.6. Xử lý sự cố thường gặp | Lỗi build nes-py, lệch API Gymnasium, CUDA, thiếu RAM | Bảng 10.4: Lỗi thường gặp và cách khắc phục | N4 | T4-N28 |

### CHƯƠNG 11. QUẢN LÝ DỰ ÁN VÀ TỔ CHỨC LÀM VIỆC NHÓM
Viết chính: **N4** · Rà soát: Cả nhóm · Hạn bản thảo: **T5-N31**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 11.1. Phân công vai trò và ma trận RACI | Vai trò 4 thành viên, RACI | Bảng 11.1: Ma trận RACI | N4 | T5-N31 |
| 11.2. Tiến độ 5 tuần | Kế hoạch so với thực tế, các mốc Sprint | Hình 11.1: Biểu đồ Gantt | N4 | T5-N31 |
| 11.3. Quy trình quản lý mã nguồn | Git Flow, Pull Request, Conventional Commits, release v1.0.0 | Hình 11.2: Sơ đồ nhánh Git | N4 | T5-N31 |
| 11.4. Quản lý rủi ro | Rủi ro đã xảy ra và cách xử lý | Bảng 11.2: Rủi ro thực tế và biện pháp | N4 | T5-N31 |
| 11.5. Đóng góp của từng thành viên | Tỷ lệ đóng góp, sản phẩm cụ thể của mỗi người | Bảng 11.3: Đóng góp thành viên | N4 (mỗi người tự khai) | T5-N31 |

### CHƯƠNG 12. KẾT LUẬN
Viết chính: **N4** · Rà soát: Cả nhóm · Hạn bản thảo: **T5-N31**

| Mục | Nội dung chính cần viết | Hình/Bảng dự kiến | Phụ trách | Hạn |
| :--- | :--- | :--- | :---: | :---: |
| 12.1. Tổng kết kết quả đạt được | Đối chiếu từng mục tiêu ở 1.3 với kết quả Ch7–8 | Bảng 12.1: Mục tiêu và mức độ hoàn thành | N4 | T5-N31 |
| 12.2. Đóng góp kỹ thuật | Pipeline tiền xử lý, agent DDQN, reward v2, bộ đánh giá/demo | | N4 | T5-N31 |
| 12.3. Bài học kinh nghiệm | Bài học kỹ thuật và làm việc nhóm | | N4 | T5-N31 |
| 12.4. Lời kết | Tóm tắt hướng phát triển (dẫn về Ch9) | | N4 | T5-N31 |

---

## 2. TÀI LIỆU THAM KHẢO DỰ KIẾN (chuẩn IEEE)

Đánh số theo thứ tự xuất hiện lần đầu trong báo cáo; danh sách dưới đây là bản dự kiến, số thứ tự sẽ được Người 4 sắp lại khi hợp nhất. Mỗi người kiểm tra lại DOI/số trang của tài liệu mình trích trước khi nộp bản thảo.

**Bài báo và sách nền tảng**

[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare *et al.*, "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015, doi: 10.1038/nature14236.

[2] H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in *Proc. 30th AAAI Conf. Artif. Intell.*, Phoenix, AZ, USA, 2016, pp. 2094–2100.

[3] Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas, "Dueling network architectures for deep reinforcement learning," in *Proc. 33rd Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 48, 2016, pp. 1995–2003.

[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

[5] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[6] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, "Playing Atari with deep reinforcement learning," arXiv:1312.5602, 2013 (NIPS Deep Learning Workshop).

[7] H. van Hasselt, "Double Q-learning," in *Adv. Neural Inf. Process. Syst. 23 (NIPS)*, 2010, pp. 2613–2621.

[8] C. J. C. H. Watkins and P. Dayan, "Q-learning," *Mach. Learn.*, vol. 8, no. 3–4, pp. 279–292, 1992.

[9] L.-J. Lin, "Self-improving reactive agents based on reinforcement learning, planning and teaching," *Mach. Learn.*, vol. 8, no. 3–4, pp. 293–321, 1992.

[10] A. Y. Ng, D. Harada, and S. Russell, "Policy invariance under reward transformations: Theory and application to reward shaping," in *Proc. 16th Int. Conf. Mach. Learn. (ICML)*, 1999, pp. 278–287.

[11] J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2016, arXiv:1506.02438.

[12] M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling, "The Arcade Learning Environment: An evaluation platform for general agents," *J. Artif. Intell. Res.*, vol. 47, pp. 253–279, 2013.

[13] M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. Hausknecht, and M. Bowling, "Revisiting the Arcade Learning Environment: Evaluation protocols and open problems for general agents," *J. Artif. Intell. Res.*, vol. 61, pp. 523–562, 2018.

[14] M. Hessel, J. Modayil, H. van Hasselt, T. Schaul, G. Ostrovski, W. Dabney *et al.*, "Rainbow: Combining improvements in deep reinforcement learning," in *Proc. 32nd AAAI Conf. Artif. Intell.*, 2018, pp. 3215–3222.

[15] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, "Prioritized experience replay," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2016, arXiv:1511.05952.

[21] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep reinforcement learning that matters," in *Proc. 32nd AAAI Conf. Artif. Intell.*, 2018, pp. 3207–3214.

[22] R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare, "Deep reinforcement learning at the edge of the statistical precipice," in *Adv. Neural Inf. Process. Syst. 34 (NeurIPS)*, 2021, pp. 29304–29320.

[23] P. J. Huber, "Robust estimation of a location parameter," *Ann. Math. Statist.*, vol. 35, no. 1, pp. 73–101, 1964.

[24] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015, arXiv:1412.6980.

**Phần mềm và thư viện**

[16] M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. De Cola, T. Deleu *et al.*, "Gymnasium: A standard interface for reinforcement learning environments," arXiv:2407.17032, 2024.

[17] C. Kauten, "Super Mario Bros for OpenAI Gym," GitHub repository, 2018. [Online]. Available: https://github.com/Kautenja/gym-super-mario-bros

[18] C. Kauten, "nes-py: An NES emulator and OpenAI Gym interface for MacOS, Linux, and Windows based on the SimpleNES emulator," GitHub repository. [Online]. Available: https://github.com/Kautenja/nes-py

[19] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. 32 (NeurIPS)*, 2019, pp. 8024–8035.

[20] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," *J. Mach. Learn. Res.*, vol. 22, no. 268, pp. 1–8, 2021.

> Chỉ trích [20] nếu PPO được cài bằng Stable-Baselines3. Các thư viện còn lại (TensorBoard, OpenCV, Pillow, NumPy) ghi trong Bảng 3.1/Bảng 6.1 kèm phiên bản thay vì đưa vào danh mục, trừ khi giảng viên yêu cầu.

---

## 3. PHỤ LỤC DỰ KIẾN

| Phụ lục | Nội dung | Người chuẩn bị | Hạn |
| :---: | :--- | :---: | :---: |
| A | Bảng siêu tham số đầy đủ và các file cấu hình (`configs/*.json`, `*.yaml`) | N3 | T5-N30 |
| B | Mã nguồn tiêu biểu: wrappers, `DuelingDQN`, `DoubleDQNAgent`, reward wrapper (trích đoạn, kèm link repo và tag `v1.0.0`) | N1, N2, N3 | T5-N30 |
| C | Kết quả chi tiết từng episode đánh giá (`results/eval_results.csv`) | N4 | T5-N30 |
| D | Số liệu thô các thí nghiệm ablation (`results/ablation_data.csv`) và đồ thị bổ sung | N3 | T5-N30 |
| E | Kết quả chạy pytest và báo cáo baseline JSON | N4, N1 | T5-N30 |
| F | Link video demo MP4, link tải checkpoint dự phòng (Google Drive) | N4 | T5-N30 |
| G | Biên bản họp Sprint và bảng phân công công việc (WBS) | N4 | T5-N31 |

---

## 4. BẢNG TỔNG HỢP PHÂN CÔNG

| Chương | Tên chương | Người viết chính | Người review | Hạn bản thảo |
| :---: | :--- | :---: | :---: | :---: |
| Mở đầu | Bìa, Lời cam đoan, Lời cảm ơn, Tóm tắt, Mục lục, Danh mục | N4 | Cả nhóm | T5-N31 |
| 1 | Giới thiệu đề tài và mục tiêu nghiên cứu | N1 | N4 | T2-N14 |
| 2 | Cơ sở lý thuyết về học tăng cường | N2 | N3 | T2-N14 |
| 3 | Phân tích bài toán Mario dưới góc nhìn RL | N1 | N2 | T2-N14 |
| 4 | Thiết kế hệ thống và kiến trúc mô hình | N2 | N1 | T3-N21 |
| 5 | Kỹ thuật thiết kế hàm phần thưởng | N3 | N2 | T3-N21 |
| 6 | Thiết lập huấn luyện và quy trình thực nghiệm | N3 | N4 | T3-N21 |
| 7 | Kết quả thực nghiệm giải màn World 1-1 | N3 | N4 | T4-N28 |
| 8 | Nghiên cứu mở rộng và ablation study | N2, N3 | N4 | T4-N28 |
| 9 | Hạn chế và hướng phát triển | N1 | N3 | T4-N28 |
| 10 | Hướng dẫn cài đặt, sử dụng và kiểm thử tái lập | N4 | N1 | T4-N28 (hoàn chỉnh T5-N30) |
| 11 | Quản lý dự án và tổ chức làm việc nhóm | N4 | Cả nhóm | T5-N31 |
| 12 | Kết luận | N4 | Cả nhóm | T5-N31 |
| TLTK, Phụ lục | Tài liệu tham khảo IEEE, Phụ lục A–G | N4 (mỗi người gửi TLTK của chương mình) | Cả nhóm | T5-N31 |
| Bản hợp nhất | Ghép chương, định dạng, mục lục tự động, xuất PDF/DOCX | N4 | Cả nhóm (rà soát N31) | T5-N32 |

Tổng khối lượng viết chính: N1 – Ch1, 3, 9; N2 – Ch2, 4, 8 (phần PPO); N3 – Ch5, 6, 7, 8 (phần ablation); N4 – Mở đầu, Ch10, 11, 12, TLTK, Phụ lục, hợp nhất.

### Ghi chú đối chiếu tài liệu kế hoạch

1. Phân công và người review lấy theo **bảng mục 7 của kế hoạch tổng thể**, khớp với sổ tay N1 (Ch1, 3, 9), N2 (Ch2, 4, 8), N3 (Ch5, 6, 7). WBS #11/#16/#21 chỉ ghi nhóm người theo đợt (Ch1–3: N1, 2; Ch4–6: N2, 3; Ch7–9: N3, 4) nên không mâu thuẫn, trừ Ch9: WBS #21 ghi "N3, 4" nhưng bảng mục 7 và sổ tay N1 giao cho N1 → theo bảng mục 7. Ma trận RACI (mục 4.2) ghi N1 viết Ch4 và N3 viết Ch6–8, không khớp bảng mục 7 → không áp dụng.
2. Sổ tay N2 ghi soạn Ch4 ở Tuần 2; bảng mục 7 đặt hạn Ch4 là T3-N21 → lấy hạn T3-N21, N2 có thể nộp sớm.
3. Ch10: bảng mục 7 đặt hạn T5-N30, sổ tay N4 đặt viết ở Tuần 4 → bản thảo T4-N28, hoàn chỉnh T5-N30.
4. Bảng mục 7 chỉ liệt kê Ch1–10 và mục "Kết luận & TLTK", trong khi kế hoạch và sổ tay N4 yêu cầu "12 chương". **Ch11 (Quản lý dự án) là đề xuất của dàn ý này** để đủ 12 chương, dựa trên mục 4, 6, 11 của kế hoạch; Ch12 là phần "Kết luận" của bảng mục 7. Cần cả nhóm xác nhận tại Sprint 1.
5. Kế hoạch mô tả đánh giá "20 seed ngẫu nhiên", nhưng `docs/ENVIRONMENT.md` lưu ý World 1-1 khởi tạo tất định, đổi seed không tạo ra lượt chơi khác. Ch6 (6.6), Ch7 và Ch9 phải nêu rõ giao thức đánh giá thực tế được dùng.

---

## 5. TÓM TẮT QUY CHUẨN ĐỊNH DẠNG (theo sổ tay Người 4, mục IV)

| Hạng mục | Quy chuẩn |
| :--- | :--- |
| Khổ giấy | A4 (210 × 297 mm) |
| Lề | Trên 2.5 cm, Dưới 2.5 cm, Trái 3.0 cm (đóng gáy), Phải 2.0 cm |
| Font | Times New Roman cho toàn bộ văn bản |
| Nội dung | 13 pt, giãn dòng 1.3–1.5, thụt đầu dòng 1.0 cm, căn đều hai bên |
| Heading 1 (Chương) | 18 pt, đậm, VIẾT HOA |
| Heading 2 (Mục x.y) | 15 pt, đậm |
| Heading 3 (Tiểu mục x.y.z) | 13 pt, đậm hoặc nghiêng |
| Hình vẽ | Chú thích **bên dưới** hình, đánh số theo chương: *Hình 4.2: Kiến trúc mạng Dueling Convolutional Neural Network* |
| Bảng biểu | Chú thích **bên trên** bảng, đánh số theo chương: *Bảng 7.1: Bảng kết quả đánh giá thực nghiệm trên 20 seeds độc lập* |
| Trích dẫn | Chuẩn **IEEE**, số trong ngoặc vuông `[1]`, `[2]`, đánh số theo thứ tự xuất hiện; tối thiểu 10 tài liệu khoa học |
| Mục lục, danh mục hình/bảng | Tạo tự động từ heading và caption |
| Công thức | Đánh số theo chương ở lề phải, vd (2.1); ký hiệu thống nhất toàn báo cáo (θ online, θ⁻ target, γ, ε) |
| Đồ thị | Xuất từ TensorBoard sang ảnh vector/độ phân giải cao, có nhãn trục và đơn vị |

Quy ước nộp bản thảo: mỗi chương là một file `docs/chapterN.md`, hình đặt trong `docs/figures/`, gửi kèm danh sách tài liệu trích dẫn và thuật ngữ viết tắt mới; commit theo dạng `docs: add chapter N draft`.
