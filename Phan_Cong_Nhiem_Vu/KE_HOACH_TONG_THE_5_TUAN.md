# KẾ HOẠCH TỔNG THỂ DỰ ÁN HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)
## ĐỀ TÀI: GIẢI QUYẾT TRÒ CHƠI SUPER MARIO BROS (WORLD 1-1) VỚI GYMNASIUM VÀ PYTHON

---

| Thông tin dự án | Chi tiết |
| :--- | :--- |
| **Môn học** | Học tăng cường (Reinforcement Learning - RL) |
| **Đề tài** | Giải trò chơi Super Mario Bros bằng Deep Reinforcement Learning |
| **Môi trường** | `gym-super-mario-bros` tích hợp `Gymnasium` (Python 3.10/3.11/3.13) |
| **Quy mô nhóm** | 04 thành viên |
| **Thời lượng thực hiện** | 05 tuần (35 ngày làm việc liên tục) |
| **Mục tiêu điểm số** | **10.0/10.0** (Phần 1: 8.0 điểm mục tiêu chính + Phần 2: 2.0 điểm cải tiến mở rộng) |

---

## MỤC LỤC TỔNG QUAN

1. [Tổng quan dự án và Phạm vi](#1-tổng-quan-dự-án-và-phạm-vi)
2. [Kiến trúc kỹ thuật và Phương pháp tiếp cận](#2-kiến-trúc-kỹ-thuật-và-phương-pháp-tiếp-cận)
3. [Phân chia vai trò 4 thành viên](#3-phân-chia-vai-trò-4-thành-viên)
4. [Bảng phân công công việc tổng hợp (WBS) & Ma trận RACI](#4-bảng-phân-công-công-việc-tổng-hợp-wbs--ma-trận-raci)
5. [Lộ trình triển khai chi tiết 5 tuần (Timeline)](#5-lộ-trình-triển-khai-chi-tiết-5-tuần-timeline)
6. [Quy trình quản lý mã nguồn và Phối hợp nhóm](#6-quy-trình-quản-lý-mã-nguồn-và-phối-hợp-nhóm)
7. [Kế hoạch thực hiện Báo cáo học thuật](#7-kế-hoạch-thực-hiện-báo-cáo-học-thuật)
8. [Kế hoạch xây dựng Slide và Diễn tập thuyết trình](#8-kế-hoạch-xây-dựng-slide-và-diễn-tập-thuyết-trình)
9. [Bộ câu hỏi và Đáp án ôn tập vấn đáp chuyên sâu](#9-bộ-câu-hỏi-và-đáp-án-ôn-tập-vấn-đáp-chuyên-sâu)
10. [Tiêu chí kiểm tra và Nghiệm thu sản phẩm](#10-tiêu-chí-kiểm-tra-và-nghiệm-thu-sản-phẩm)
11. [Bảng quản lý rủi ro và Kế hoạch ứng phó](#11-bảng-quản-lý-rủi-ro-và-kế-hoạch-ứng-phó)
12. [Checklist hoàn thành dự án toàn diện](#12-checklist-hoàn-thành-dự-án-toàn-diện)

---

## 1. TỔNG QUAN DỰ ÁN VÀ PHẠM VI

### 1.1. Bối cảnh đề tài
Trò chơi **Super Mario Bros (NES, 1985)** là một trong những môi trường benchmark kinh điển nhưng đầy thách thức trong lĩnh vực Deep Reinforcement Learning (DRL). Khác với các trò chơi Atari đơn giản (Pong, Breakout), Super Mario Bros sở hữu:
- **Không gian quan sát trực quan phức tạp:** Khung hình chuyển động cuộn ngang (side-scrolling), nhiều kẻ thù di động (Goomba, Koopa Troopa), chướng ngại vật tĩnh (ống nước, hố sâu) và các khối gạch tương tác.
- **Vấn đề phần thưởng trễ (Delayed Rewards) và Thưa thớt (Sparse Rewards):** Mục tiêu tối thượng là chạm cờ (Flagpole) ở cuối màn đòi hỏi hàng ngàn bước hành động chính xác liên tiếp mà không được chết.
- **Hành vi đòi hỏi sự phối hợp chuỗi hành động nhạy bén (Temporal Dependency):** Việc nhảy qua hố sâu hoặc vượt ống nước cao đòi hỏi nhấn giữ tổ hợp phím (Right + B để chạy nhanh lấy đà, sau đó nhấn A để nhảy cao).

### 1.2. Phân chia cấu trúc điểm số
- **Phần 1 – Mục tiêu chính (8.0 điểm):**
  - Xây dựng thành công tác nhân AI (Agent) bằng thuật toán DRL có khả năng điều khiển Mario tự động vượt qua toàn bộ màn chơi **World 1-1** (chạm cột cờ `flag_get == True`).
  - Đánh giá trên tập kiểm thử độc lập (ít nhất 20 episodes với các seed ngẫu nhiên), đạt tỷ lệ hoàn thành màn chơi (Win Rate) $\ge 80\%$.
  - Toàn bộ pipeline từ tiền xử lý môi trường, kiến trúc mạng, replay buffer đến vòng lặp huấn luyện/đánh giá hoạt động trơn tru.
- **Phần 2 – Cải tiến và Mở rộng (2.0 điểm):**
  - Nghiên cứu và triển khai **Nghiên cứu triệt tiêu (Ablation Study)** so sánh:
    - *Thuật toán:* So sánh đối sánh hiệu năng giữa **Dueling Double DQN** và **PPO (Proximal Policy Optimization)**.
    - *Reward Function:* So sánh giữa *Sparse/Default Reward* và *Custom Shaped Reward* (thưởng vận tốc, phạt dậm chân tại chỗ, thưởng chạm cờ).
    - *Không gian hành động:* So sánh `RIGHT_ONLY` (5 hành động) vs `SIMPLE_MOVEMENT` (7 hành động).
  - Xây dựng dashboard trực quan hóa quá trình huấn luyện bằng TensorBoard và công cụ tự động xuất video MP4 benchmark chất lượng cao.

---

## 2. KIẾN TRÚC KỸ THUẬT VÀ PHƯƠNG PHÁP TIẾP CẬN

### 2.1. Mô hình hóa bài toán dưới dạng Markov Decision Process (MDP)
Bài toán Super Mario Bros được mô hình hóa qua bộ 5 thành phần $(S, A, P, R, \gamma)$:
1. **Không gian trạng thái ($S$):** Khung hình gốc của NES là $240 \times 256 \times 3$ (RGB). Sau khi qua pipeline tiền xử lý:
   - Chuyển sang ảnh xám (Grayscale): $240 \times 256 \times 1$.
   - Thu nhỏ kích thước (Downsampling): $84 \times 84$.
   - Bỏ qua khung hình (Frame Skip = 4): Tác nhân chỉ ra quyết định mỗi 4 frame, giữ nguyên hành động trong 4 frame đó để giảm tải tính toán và khớp với tốc độ phản ứng vật lý.
   - Xếp chồng khung hình (Frame Stacking = 4): Ghép 4 frame xám liên tiếp thành tensor kích thước $(4, 84, 84)$. Việc xếp chồng giúp mạng nơ-ron nhận biết được vận tốc, gia tốc và chiều chuyển động của Mario cũng như quái vật.
2. **Không gian hành động ($A$):** Rút gọn từ 256 tổ hợp nút bấm NES về `SIMPLE_MOVEMENT` gồm 7 hành động rời rạc:
   - `0`: NOOP (Không làm gì)
   - `1`: Đi sang phải (Right)
   - `2`: Nhảy sang phải (Right + A)
   - `3`: Chạy nhanh sang phải (Right + B)
   - `4`: Chạy nhanh và nhảy sang phải (Right + A + B)
   - `5`: Nhảy thẳng đứng (A)
   - `6`: Đi sang trái (Left)
3. **Hàm phần thưởng ($R$ - Reward Function):**
   $$R_t = R_{\text{progress}} + R_{\text{time}} + R_{\text{death}} + R_{\text{flag}} + R_{\text{status}}$$
   - $R_{\text{progress}} = \text{clip}(x_t - x_{t-1}, -5, 5)$: Thưởng cho việc di chuyển tiến về bên phải theo trục X.
   - $R_{\text{time}} = -0.1$: Phạt thời gian trên mỗi bước để khuyến khích Mario về đích nhanh nhất, tránh đi lang thang.
   - $R_{\text{death}} = -50$: Phạt nặng khi Mario rơi xuống hố hoặc va chạm quái vật làm mất mạng.
   - $R_{\text{flag}} = +500$: Thưởng cực lớn khi kéo cờ chiến thắng (`info['flag_get'] == True`).
   - $R_{\text{stuck}} = -1.0$: Phạt nếu Mario đứng yên một chỗ quá 50 bước liên tiếp.
4. **Hệ số chiết khấu ($\gamma$):** $\gamma = 0.99$, thể hiện tầm nhìn dài hạn của tác nhân hướng tới cột cờ cuối màn.

### 2.2. Lựa chọn thuật toán chính và Thuật toán đối sánh
- **Thuật toán chính (Primary - Mục tiêu 8 điểm): Dueling Double Deep Q-Network (Dueling DDQN)**
  - *Lý do lựa chọn:* Bản chất của Mario là không gian hành động rời rạc (7 actions). DQN truyền thống thường bị phóng đại giá trị Q (Overestimation Bias). Double DQN giải quyết triệt để vấn đề này bằng cách tách rời việc chọn hành động (dùng Online Network) và đánh giá giá trị hành động (dùng Target Network):
    $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$
  - Kiến trúc *Dueling Network* tách biệt biểu diễn trạng thái thành hai luồng: Luồng giá trị trạng thái $V(s)$ và Luồng ưu thế của từng hành động $A(s, a)$, giúp mạng học nhanh trạng thái nào là an toàn/nguy hiểm bất kể hành động nào được chọn:
    $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|A|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$
- **Thuật toán đối sánh (Ablation/Extension - 2 điểm cộng): PPO (Proximal Policy Optimization)**
  - Thuật toán On-Policy dạng Actor-Critic với hàm mục tiêu Clipped Surrogate Objective, đại diện cho trường phái Policy Gradient hiện đại, ổn định và kháng biến động gradient tốt.

---

## 3. PHÂN CHIA VAI TRÒ 4 THÀNH VIÊN

Nhóm gồm 4 thành viên, mỗi người chịu trách nhiệm chính (Owner) trên một trụ cột kỹ thuật, đồng thời tham gia chéo vào các công tác thử nghiệm và học thuật:

```
+-----------------------------------------------------------------------------------+
|                           BAN ĐIỀU PHỐI DỰ ÁN MARIO RL                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [NGƯỜI 1]                      [NGƯỜI 2]                      [NGƯỜI 3]         |
|  Kỹ sư Môi trường &             Kỹ sư Thuật toán DRL            Kỹ sư Huấn luyện, |
|  Tiền xử lý dữ liệu             & Kiến trúc Mạng                Reward & Thí nghiệm|
|  - Môi trường Gymnasium         - Xây dựng Dueling DDQN         - Thiết kế Reward |
|  - FrameSkip, FrameStack        - Replay Buffer, Epsilon        - Hyperparameter  |
|  - Action Space Wrapper         - Target Network update         - TensorBoard log |
|  - Baseline kiểm thử            - PyTorch Network Pipeline      - Nghiên cứu triệt tiêu|
|                                                                                   |
|                                         |                                         |
|                                         v                                         |
|                                   [NGƯỜI 4]                                       |
|                       Kỹ sư Tích hợp Hệ thống, Kiểm thử,                         |
|                       Đánh giá Benchmark & Trưởng nhóm Học thuật                  |
|                       - Cấu trúc Git, CI/CD, Makefile/Shell                      |
|                       - Tích hợp Module, Pytest, Deterministic Eval               |
|                       - Render Video demo MP4, Trực quan hóa                      |
|                       - Điều phối Báo cáo học thuật & Slide thuyết trình          |
+-----------------------------------------------------------------------------------+
```

### Chi tiết phân vai:
1. **Người 1 – Chuyên gia Môi trường & Tiền xử lý (Environment & Preprocessing Lead):**
   - Chịu trách nhiệm thiết lập môi trường giả lập NES Mario tương thích 100% chuẩn Gymnasium mới nhất.
   - Xây dựng toàn bộ các Custom Wrapper: Chuyển ảnh xám, Resize $84 \times 84$, Frame Skip 4, Frame Stack 4, Joypad Action Space.
   - Cung cấp hàm đóng gói `make_mario_env()` chuẩn hóa cho toàn đội.
2. **Người 2 – Chuyên gia Thuật toán Deep RL (Algorithm & Deep Learning Lead):**
   - Chịu trách nhiệm xây dựng kiến trúc mạng nơ-ron tích chập (Nature CNN + Dueling Head) bằng PyTorch.
   - Cài đặt thuật toán Dueling Double DQN, cơ chế Epsilon-Greedy Decay, Replay Buffer lưu trữ $100,000$ transitions, cập nhật mạng Target định kỳ.
   - Cài đặt pipeline lưu/tải mô hình (`save_checkpoint`, `load_checkpoint`).
3. **Người 3 – Chuyên gia Huấn luyện, Reward & Thực nghiệm (Training & Experimentation Lead):**
   - Thiết kế và tinh chỉnh hàm phần thưởng (Reward Shaping) giải quyết dứt điểm các bẫy tâm lý của Mario (sợ quái vật, nhảy lùi, kẹt ở ống nước).
   - Thiết lập hệ thống giám sát thời gian thực với TensorBoard (Reward, Episode Length, Max X-Position, Loss, Epsilon).
   - Thực hiện huấn luyện mô hình chính và chủ trì các thí nghiệm đối sánh (Ablation Studies).
4. **Người 4 – Chuyên gia Tích hợp, Đánh giá, Trực quan hóa & Báo cáo (Integration, QA & Lead Coordinator):**
   - Thiết lập cấu trúc Repository chuẩn quốc tế, quy ước Git Flow, quản trị mã nguồn.
   - Xây dựng module đánh giá định lượng độc lập (`evaluate.py`) với các seed cố định, trích xuất Win Rate, X-pos trung bình.
   - Xây dựng công cụ ghi video tự động (`demo.py`, xuất file MP4 chất lượng cao).
   - Tổng hợp, biên tập và chuẩn hóa bản báo cáo khoa học 12 chương, thiết kế Slide và tổ chức các buổi diễn tập vấn đáp.

---

## 4. BẢNG PHÂN CÔNG CÔNG VIỆC TỔNG HỢP (WBS) & MA TRẬN RACI

### 4.1. Bảng phân công chi tiết theo mức độ ưu tiên
- **P0 (Critical):** Nhiệm vụ sống còn bắt buộc phải hoàn thành để đạt mục tiêu 8.0 điểm (Mario thắng World 1-1).
- **P1 (High):** Nhiệm vụ hoàn thiện sản phẩm kỹ thuật, đảm bảo tính ổn định, tái lập và báo cáo chuẩn mực.
- **P2 (Medium/Extension):** Nhiệm vụ cải tiến nâng cao phục vụ lấy điểm tối đa 10.0 (Ablation study, đối sánh thuật toán, video demo đẹp).

| STT | Gói công việc (WBS) | Người phụ trách chính | Người phối hợp | Thời gian | Sản phẩm đầu ra (Deliverables) | Mức ưu tiên |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **1** | Thiết lập môi trường ảo, tương thích Gymnasium, xử lý lỗi nes-py trên Windows/Linux | **Người 1** | Người 4 | Tuần 1 (Ngày 1-3) | `environment.yml`, `requirements.txt`, tài liệu cài đặt | **P0** |
| **2** | Khởi tạo cấu trúc Repo, Git branching, quy ước commit và CI testing sơ bộ | **Người 4** | Cả nhóm | Tuần 1 (Ngày 1-2) | Khung repository chuẩn, `.gitignore`, `README.md` | **P0** |
| **3** | Xây dựng bộ Wrapper tiền xử lý (Grayscale, Resize 84x84, Skip 4, Stack 4, Joypad) | **Người 1** | Người 2 | Tuần 1 (Ngày 3-5) | `src/env/wrappers.py`, `src/env/make_env.py` | **P0** |
| **4** | Thiết kế kiến trúc CNN Feature Extractor & Dueling Q-Network (PyTorch) | **Người 2** | Người 1 | Tuần 1 (Ngày 4-7) | `src/models/dqn_network.py`, unit test tensor shapes | **P0** |
| **5** | Xây dựng script kiểm tra môi trường baseline với Random Agent (đo FPS, memory) | **Người 1** | Người 3 | Tuần 1 (Ngày 6-7) | `tests/test_env.py`, `baseline_random.py` | **P0** |
| **6** | Lập dàn ý chi tiết Báo cáo và phân công các mục lý thuyết | **Người 4** | Cả nhóm | Tuần 1 (Ngày 6-7) | Đề cương báo cáo Word/Markdown chi tiết | **P1** |
| **7** | Triển khai Replay Buffer, Loss Huber, Target Network và thuật toán Double DQN | **Người 2** | Người 3 | Tuần 2 (Ngày 8-11) | `src/agents/double_dqn.py`, `src/utils/replay_buffer.py` | **P0** |
| **8** | Thiết kế Reward Function v1 (Reward vị trí X, phạt chết, phạt thời gian) | **Người 3** | Người 1 | Tuần 2 (Ngày 8-11) | `src/rewards/custom_reward.py` | **P0** |
| **9** | Xây dựng pipeline huấn luyện hoàn chỉnh (`train.py`), lưu checkpoint và TensorBoard | **Người 2** | Người 3 | Tuần 2 (Ngày 11-13) | `train.py`, `src/utils/logger.py` | **P0** |
| **10** | Tích hợp sơ bộ toàn hệ thống, chạy thử nghiệm 50,000 steps kiểm tra rò rỉ RAM | **Người 4** | Người 2 | Tuần 2 (Ngày 13-14) | Checkpoint ban đầu, xác nhận hệ thống chạy thông suốt | **P0** |
| **11** | Soạn thảo Chương 1, 2, 3 của Báo cáo học thuật (Tổng quan, Lý thuyết RL, MDP Mario) | Người 1, 2 | Người 4 | Tuần 2 (Ngày 12-14) | Bản thảo Chương 1-3 | **P1** |
| **12** | Huấn luyện chuyên sâu mô hình chính (Dueling DDQN) từ 1.0M đến 3.0M steps | **Người 3** | Người 2 | Tuần 3 (Ngày 15-18) | Checkpoint `best_model.pth`, log TensorBoard | **P0** |
| **13** | Phân tích hành vi Mario: khắc phục bẫy kẹt ống nước, sợ Goomba, rơi xuống hố | **Người 3** | Người 1, 2 | Tuần 3 (Ngày 17-19) | Phiên bản Reward v2 (Anti-stuck, Jump incentive) | **P0** |
| **14** | Xây dựng bộ công cụ đánh giá độc lập (`evaluate.py`) với 20 seeds ngẫu nhiên | **Người 4** | Người 3 | Tuần 3 (Ngày 18-20) | `evaluate.py`, ma trận kết quả (Win Rate, Max X) | **P0** |
| **15** | Đạt mốc Mario chạm cờ World 1-1 (`flag_get == True`) và trích xuất bằng chứng | Cả nhóm | Cả nhóm | Tuần 3 (Ngày 20-21) | Video Mario thắng màn 1-1, log nghiệm thu P0 | **P0** |
| **16** | Soạn thảo Chương 4, 5, 6 của Báo cáo (Thiết kế hệ thống, Thuật toán, Quá trình train) | Người 2, 3 | Người 4 | Tuần 3 (Ngày 19-21) | Bản thảo Chương 4-6 | **P1** |
| **17** | Triển khai thí nghiệm mở rộng 1: Đối sánh thuật toán PPO vs Dueling DDQN (2 điểm cộng) | **Người 2** | Người 3 | Tuần 4 (Ngày 22-25) | Module PPO, bảng so sánh tốc độ hội tụ | **P2** |
| **18** | Triển khai thí nghiệm mở rộng 2: Nghiên cứu triệt tiêu Reward Shaping và Action Space | **Người 3** | Người 1 | Tuần 4 (Ngày 24-27) | Biểu đồ so sánh các biến thể trên TensorBoard | **P2** |
| **19** | Hoàn thiện module trực quan hóa và tự động xuất video demo MP4 full-HD kèm overlay | **Người 4** | Người 1 | Tuần 4 (Ngày 25-27) | `demo.py`, video demo có chèn thông số thời gian thực | **P1** |
| **20** | Thử nghiệm cài đặt sạch trên máy tính trắng (Reproducibility Testing) | **Người 4** | Người 1 | Tuần 4 (Ngày 27-28) | Báo cáo kiểm thử cài đặt độc lập thành công | **P1** |
| **21** | Soạn thảo Chương 7, 8, 9 của Báo cáo (Kết quả thực nghiệm, Cải tiến mở rộng, Thảo luận) | Người 3, 4 | Cả nhóm | Tuần 4 (Ngày 26-28) | Bản thảo Chương 7-9 | **P1** |
| **22** | Đóng băng toàn bộ mã nguồn (Code Freeze), dọn dẹp refactor và viết docstrings | **Người 4** | Cả nhóm | Tuần 5 (Ngày 29-30) | Release v1.0.0 trên GitHub, mã nguồn sạch đẹp | **P0** |
| **23** | Hoàn thiện toàn diện Báo cáo học thuật (Chương 1-12, Tài liệu tham khảo, Format chuẩn) | **Người 4** | Cả nhóm | Tuần 5 (Ngày 30-32) | File Báo cáo đồ án hoàn chỉnh (PDF & DOCX) | **P0** |
| **24** | Thiết kế bộ Slide thuyết trình chuyên nghiệp (tối đa 25-30 slides, ưu tiên đồ thị, video) | **Người 4** | Cả nhóm | Tuần 5 (Ngày 31-33) | File Slide thuyết trình chính thức | **P0** |
| **25** | Biên soạn Bộ câu hỏi và đáp án vấn đáp lý thuyết & thực hành chuyên sâu | Cả nhóm | Cả nhóm | Tuần 5 (Ngày 32-34) | Tài liệu ôn tập bảo vệ đồ án | **P0** |
| **26** | Tổ chức 02 buổi diễn tập thuyết trình và phản biện thử nghiệm (Mock Defense) | Cả nhóm | Cả nhóm | Tuần 5 (Ngày 34-35) | Ghi nhận phản hồi, điều chỉnh phân chia lời thoại | **P0** |

---

### 4.2. Ma trận RACI (Responsible, Accountable, Consulted, Informed)
- **R (Responsible):** Người trực tiếp thực hiện công việc.
- **A (Accountable):** Người chịu trách nhiệm cuối cùng về chất lượng và tiến độ của đầu ra (chỉ duy nhất 1 người).
- **C (Consulted):** Người được hỏi ý kiến chuyên môn và hỗ trợ kỹ thuật.
- **I (Informed):** Người được thông báo kết quả sau khi hoàn thành.

| Khối công việc | Người 1 (Môi trường) | Người 2 (Thuật toán) | Người 3 (Huấn luyện) | Người 4 (Tích hợp & QA) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Môi trường & Preprocessing Wrappers** | **A / R** | C | C | I |
| **2. Kiến trúc Mạng & Thuật toán DDQN** | C | **A / R** | C | I |
| **3. Replay Buffer & Training Engine** | I | **A / R** | R | I |
| **4. Reward Engineering & Khắc phục bẫy hành vi** | C | C | **A / R** | I |
| **5. Huấn luyện thực tế & Tinh chỉnh Hyperparameters** | I | C | **A / R** | I |
| **6. Quản trị Repository, Branching & Git Flow** | I | I | I | **A / R** |
| **7. Đánh giá Benchmark độc lập & Xuất Video MP4** | I | I | C | **A / R** |
| **8. Triển khai Cải tiến 2 điểm (Ablation / PPO)** | C | R | R | **A** |
| **9. Tổng hợp Báo cáo học thuật cuối kỳ** | R (Ch.1,3,4) | R (Ch.2,5) | R (Ch.6,7,8) | **A / R** (Tổng tài) |
| **10. Thiết kế Slide & Diễn tập bảo vệ** | R | R | R | **A / R** |

---

## 5. LỘ TRÌNH TRIỂN KHAI CHI TIẾT 5 TUẦN (TIMELINE)

### TUẦN 1 (NGÀY 1 – NGÀY 7): NGHIÊN CỨU, THIẾT LẬP NỀN TẢNG VÀ THIẾT KẾ HỆ THỐNG
- **Mục tiêu tuần 1:** Khắc phục triệt để tương thích thư viện NES Mario với Gymnasium; cả 4 thành viên cài đặt thành công trên máy cá nhân; kiến trúc mạng CNN và quy ước Git được thống nhất hoàn toàn.
- **Nhiệm vụ cụ thể từng ngày:**
  - *Ngày 1:* Người 4 tạo Git repository, cấu hình `.gitignore`, thiết lập quy ước branch. Người 1 khảo sát các phiên bản `gym-super-mario-bros` và `nes-py`, xác định xung đột trên Python 3.12/3.13 và giải pháp môi trường ổn định (Python 3.10/3.11 hoặc shim wrapper cho Gymnasium).
  - *Ngày 2:* Người 1 hoàn thành file cấu hình môi trường chuẩn (`requirements.txt`, `environment.yml`). Cả 4 thành viên cài đặt thử nghiệm và chạy lệnh test khởi tạo màn hình game Mario.
  - *Ngày 3:* Người 1 hoàn thành các Wrapper: Grayscale, Resize $84 \times 84$. Người 2 nghiên cứu cấu trúc Nature CNN (Mnih et al., 2015) và Dueling architecture.
  - *Ngày 4:* Người 1 hoàn thành FrameSkipWrapper (skip 4) và FrameStackWrapper (stack 4). Người 2 lập trình khung class `DuelingDQN` trong PyTorch.
  - *Ngày 5:* Người 1 tích hợp toàn bộ vào hàm `make_mario_env()`. Người 2 viết unit test kiểm tra tensor shape đầu vào `(Batch, 4, 84, 84)` và đầu ra `(Batch, 7)`.
  - *Ngày 6:* Người 1 và Người 3 chạy script `baseline_random.py` để đo FPS render, mức chiếm dụng RAM và VRAM. Người 4 lập dàn ý chi tiết mục lục Báo cáo đồ án.
  - *Ngày 7:* Họp Sprint 1 tổng kết: Nghiệm thu môi trường chạy mượt mà, thông qua tài liệu kiến trúc và phân chia mục tiêu tuần 2.
- **Sản phẩm đầu ra tuần 1:**
  1. `requirements.txt` và tài liệu hướng dẫn cài đặt môi trường.
  2. Module tiền xử lý: `src/env/wrappers.py`, `src/env/make_env.py`.
  3. Kiến trúc mạng: `src/models/dqn_network.py`.
  4. Script baseline: `tests/test_env.py`, `baseline_random.py`.
  5. Dàn ý Báo cáo đồ án chi tiết (Word/Markdown).

---

### TUẦN 2 (NGÀY 8 – NGÀY 14): TRIỂN KHAI PHIÊN BẢN THUẬT TOÁN ĐẦU TIÊN (ALPHA PIPELINE)
- **Mục tiêu tuần 2:** Xây dựng hoàn chỉnh thuật toán Dueling Double DQN, Replay Buffer và vòng lặp huấn luyện; tác nhân bắt đầu tương tác có định hướng với game; hệ thống log TensorBoard hoạt động ổn định.
- **Nhiệm vụ cụ thể từng ngày:**
  - *Ngày 8:* Người 2 triển khai `ReplayBuffer` với dung lượng $100,000$ transitions (sử dụng deque hoặc numpy array tối ưu bộ nhớ). Người 3 thiết kế hàm phần thưởng ban đầu (Reward v1).
  - *Ngày 9:* Người 2 xây dựng class `DoubleDQNAgent`: hàm chọn hành động $\epsilon$-greedy, hàm tính toán loss Huber giữa Q-online và Q-target.
  - *Ngày 10:* Người 2 triển khai cơ chế cập nhật mạng đích (Target Network update soft/hard mỗi $10,000$ steps). Người 3 thiết lập class `TensorboardLogger` ghi nhận: `loss`, `epsilon`, `mean_reward`, `x_pos`.
  - *Ngày 11:* Người 2 và Người 3 kết nối toàn bộ các module thành script `train.py`. Chạy thử nghiệm $10,000$ steps đầu tiên để bắt lỗi logic và ngoại lệ runtime.
  - *Ngày 12:* Người 4 thiết lập cơ chế kiểm thử tích hợp tự động, kiểm tra tính tương thích giữa output của môi trường (Người 1) và input của mạng nơ-ron (Người 2). Người 1, 2 bắt đầu viết bản thảo Chương 1, 2, 3 của báo cáo.
  - *Ngày 13:* Người 3 tiến hành chạy thử nghiệm $100,000$ steps qua đêm. Theo dõi hiện tượng rò rỉ bộ nhớ (Memory Leak) và tốc độ huấn luyện (steps/second).
  - *Ngày 14:* Họp Sprint 2: Phân tích đường cong học tập ban đầu. Đảm bảo Mario đã biết học cách nhấn phím tiến về bên phải thay vì đứng yên hoặc chỉ nhảy tại chỗ.
- **Sản phẩm đầu ra tuần 2:**
  1. Module thuật toán: `src/agents/double_dqn.py`, `src/utils/replay_buffer.py`.
  2. Module phần thưởng ban đầu: `src/rewards/custom_reward.py`.
  3. Script huấn luyện hoàn chỉnh: `train.py`.
  4. Hệ thống log và checkpoint: Thư mục `logs/` và `checkpoints/`.
  5. Bản thảo Chương 1, 2, 3 Báo cáo học thuật.

---

### TUẦN 3 (NGÀY 15 – NGÀY 21): HUẤN LUYỆN CHUYÊN SÂU & HOÀN THÀNH MỤC TIÊU 8 ĐIỂM (WORLD 1-1)
- **Mục tiêu tuần 3:** **MỐC QUAN TRỌNG NHẤT DỰ ÁN.** Huấn luyện tác nhân hoàn thành xuất sắc màn World 1-1 (chạm cờ `flag_get == True`), tỷ lệ thắng $\ge 80\%$ trên tập test; trích xuất toàn bộ bằng chứng định lượng.
- **Nhiệm vụ cụ thể từng ngày:**
  - *Ngày 15:* Người 3 cấu hình siêu tham số tối ưu: Learning rate $= 1\times 10^{-4}$ (Adam), Batch size $= 64$, Gamma $= 0.99$, Epsilon decay từ $1.0$ về $0.05$ trong $1.0\times 10^6$ steps. Bắt đầu đợt huấn luyện lớn 3.0 triệu steps.
  - *Ngày 16:* Người 3 và Người 2 trực monitor tiến độ. Phát hiện các điểm nghẽn hành vi: Mario kẹt ở ống nước thứ nhất hoặc nhảy vào người Goomba.
  - *Ngày 17:* Người 3 tinh chỉnh Reward Function v2: Bổ sung phạt dậm chân tại chỗ quá 50 frames (Anti-stuck penalty), tăng thưởng vượt chướng ngại vật theo trục X, thưởng kích hoạt nhảy qua ống nước cao.
  - *Ngày 18:* Người 4 viết script đánh giá độc lập `evaluate.py`: chạy tác nhân ở chế độ hoàn toàn khai thác ($\epsilon = 0.0$) trên 20 seed ngẫu nhiên, ghi nhận chính xác: Win Rate, Average X-position, Max X-position, Average Steps.
  - *Ngày 19:* Mô hình đạt mốc vượt qua hố sâu thứ hai và tiến sát cột cờ ($X \approx 3000$). Tiến hành fine-tuning nhẹ tốc độ học.
  - *Ngày 20:* **KHOẢNH KHẮC MỤC TIÊU:** Checkpoint `best_model_mario_v1.pth` vượt màn 1-1 thành công, chạm cờ tại vị trí $X \ge 3161$. Người 4 chạy `evaluate.py` xác nhận tỷ lệ thắng đạt $\ge 85\%$.
  - *Ngày 21:* Họp Sprint 3: Cả nhóm xác nhận chính thức hoàn thành mục tiêu 8.0 điểm. Người 4 lưu trữ checkpoint vàng, xuất log Tensorboard và video chạy demo thành công. Người 2, 3 hoàn thiện bản thảo Chương 4, 5, 6 của báo cáo.
- **Sản phẩm đầu ra tuần 3:**
  1. Checkpoint chính thức: `checkpoints/dueling_ddqn_world1_1_best.pth`.
  2. Bằng chứng định lượng: Bảng đánh giá 20 episodes và đồ thị TensorBoard minh chứng hội tụ.
  3. Video ghi hình Mario hoàn thành màn 1-1 mượt mà.
  4. Script đánh giá benchmark: `evaluate.py`.
  5. Bản thảo Chương 4, 5, 6 Báo cáo học thuật.

---

### TUẦN 4 (NGÀY 22 – NGÀY 28): TRIỂN KHAI CẢI TIẾN MỞ RỘNG (2 ĐIỂM CỘNG) & HOÀN THIỆN KỸ THUẬT
- **Mục tiêu tuần 4:** Thực hiện các nghiên cứu đối sánh học thuật có giá trị cao; hoàn thiện module quay video demo chất lượng cao; kiểm tra tính tái lập toàn diện trên nhiều môi trường máy tính khác nhau.
- **Nhiệm vụ cụ thể từng ngày:**
  - *Ngày 22:* Người 2 triển khai thuật toán đối sánh PPO (sử dụng kiến trúc Actor-Critic tương đương hoặc Stable-Baselines3 làm baseline chuẩn) để thực hiện thí nghiệm so sánh Dueling DDQN vs PPO.
  - *Ngày 23:* Người 3 thiết lập kịch bản Ablation Study 1: So sánh hiệu quả của việc có và không có Reward Shaping (Sparse Reward vs Shaped Reward).
  - *Ngày 24:* Người 3 và Người 1 thiết lập kịch bản Ablation Study 2: So sánh không gian hành động `RIGHT_ONLY` (5 hành động) vs `SIMPLE_MOVEMENT` (7 hành động).
  - *Ngày 25:* Người 4 xây dựng script `demo.py`: Sử dụng thư viện `pygame` và OpenCV để hiển thị trực tiếp quá trình chơi của Mario, tự động ghi video MP4 sắc nét kèm bảng hiển thị HUD (Heads-up Display) các thông số: Action hiện tại, Q-values dự đoán, Epsilon, Tọa độ X, Phần thưởng tích lũy.
  - *Ngày 26:* Người 4 tiến hành kiểm thử tái lập (Reproducibility Test): Clone repo sang một máy tính mới hoàn toàn, chạy lại từ file hướng dẫn để đảm bảo không phụ thuộc vào đường dẫn cục bộ (Hard-coded path).
  - *Ngày 27:* Người 3 tổng hợp toàn bộ số liệu thực nghiệm của các thí nghiệm cải tiến thành các bảng biểu và biểu đồ so sánh chuyên nghiệp cho báo cáo.
  - *Ngày 28:* Họp Sprint 4: Đánh giá kết quả phần cải tiến 2.0 điểm. Khóa toàn bộ việc phát triển tính năng mới (Code Freeze) để tập trung 100% cho tài liệu và diễn tập bảo vệ.
- **Sản phẩm đầu ra tuần 4:**
  1. Kết quả thí nghiệm mở rộng: Bảng số liệu và đồ thị so sánh Dueling DDQN vs PPO, Ablation Reward.
  2. Module Demo & Ghi hình: `demo.py`, bộ video MP4 chất lượng cao lưu tại `recordings/`.
  3. Báo cáo kiểm thử tính tái lập môi trường độc lập.
  4. Bản thảo Chương 7, 8, 9 Báo cáo học thuật.

---

### TUẦN 5 (NGÀY 29 – NGÀY 35): KIỂM THỬ CUỐI, BÁO CÁO, SLIDE & TỔ CHỨC DIỄN TẬP VẤN ĐÁP
- **Mục tiêu tuần 5:** Đóng gói trọn vẹn sản phẩm; xuất bản Báo cáo học thuật đạt chuẩn quốc tế; Slide thuyết trình sắc nét; 4 thành viên thuộc bài và trả lời lưu loát mọi câu hỏi phản biện của hội đồng chấm thi.
- **Nhiệm vụ cụ thể từng ngày:**
  - *Ngày 29:* Người 4 chủ trì việc hợp nhất toàn bộ mã nguồn vào nhánh `main`, tạo thẻ phát hành `Release v1.0.0`. Cả 4 thành viên cùng kiểm tra lại từng dòng comment và định dạng docstrings theo chuẩn Google/PEP8.
  - *Ngày 30:* Người 4 tổng hợp các chương của báo cáo thành file Word/PDF duy nhất. Chuẩn hóa font chữ Times New Roman, cỡ chữ, căn lề, mục lục tự động, danh mục hình vẽ, bảng biểu và trích dẫn IEEE.
  - *Ngày 31:* Cả nhóm cùng rà soát lỗi chính tả, câu từ, tính chính xác của các công thức toán học và sự khớp nối giữa báo cáo và mã nguồn.
  - *Ngày 32:* Người 4 hoàn thiện bộ Slide thuyết trình (PowerPoint/Canva) gồm 25-30 slides. Phân chia rành mạch phần trình bày của 4 thành viên (mỗi người 5-7 phút).
  - *Ngày 33:* Cả nhóm cùng biên soạn Bộ tài liệu ôn tập vấn đáp chuyên sâu (Flashcards & Q&A handbook) bao gồm 20 câu hỏi lý thuyết và tình huống thực hành.
  - *Ngày 34:* Tổ chức **Buổi diễn tập bảo vệ lần 1 (Mock Defense 1)**: Bấm giờ chính xác 20 phút thuyết trình, các thành viên đóng vai hội đồng để đặt câu hỏi khó và phản biện lẫn nhau.
  - *Ngày 35:* Tổ chức **Buổi diễn tập bảo vệ lần 2 (Mock Defense 2)**: Tinh chỉnh lại phong thái, giọng điệu, slide chuyển tiếp; kiểm tra kỹ thuật máy chiếu, loa, chạy demo offline dự phòng. Sẵn sàng 100% cho buổi bảo vệ chính thức!
- **Sản phẩm đầu ra tuần 5:**
  1. Repository hoàn thiện đạt chuẩn chất lượng cao, có tài liệu README đầy đủ.
  2. Báo cáo học thuật cuối kỳ bản PDF và Word chính thức.
  3. Bộ Slide thuyết trình hoàn chỉnh.
  4. Bộ câu hỏi và đáp án vấn đáp phục vụ ôn thi.
  5. Thư mục Demo sẵn sàng chạy offline không cần Internet.

---

## 6. QUY TRÌNH QUẢN LÝ MÃ NGUỒN VÀ PHỐI HỢP NHÓM

### 6.1. Cấu trúc Repository chuẩn mực
```
RL_Mario_Solving/
├── configs/                      # Chứa các file cấu hình YAML (hyperparameters, env config)
│   ├── dqn_baseline.yaml
│   └── ppo_ablation.yaml
├── src/                          # Mã nguồn chính của dự án
│   ├── env/                      # [Người 1] Quản lý môi trường và tiền xử lý
│   │   ├── __init__.py
│   │   ├── wrappers.py           # Custom wrappers: Grayscale, Resize, FrameSkip, FrameStack
│   │   └── make_env.py           # Factory function khởi tạo vectorized environment
│   ├── models/                   # [Người 2] Kiến trúc mạng nơ-ron
│   │   ├── __init__.py
│   │   └── dqn_network.py        # Nature CNN, Dueling Architecture
│   ├── agents/                   # [Người 2] Thuật toán học tăng cường
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── double_dqn.py         # Thuật toán Dueling Double DQN chính
│   │   └── ppo_agent.py          # Thuật toán đối sánh PPO
│   ├── rewards/                  # [Người 3] Thiết kế hàm phần thưởng
│   │   ├── __init__.py
│   │   └── custom_reward.py      # Reward shaping v1, v2, anti-stuck
│   └── utils/                    # [Người 4 & Người 2] Tiện ích hỗ trợ
│       ├── __init__.py
│       ├── replay_buffer.py      # Bộ nhớ kinh nghiệm chuyển tiếp
│       └── logger.py             # Wrapper ghi nhận TensorBoard
├── tests/                        # [Người 4] Kiểm thử tự động (Unit test, Integration test)
│   ├── test_env.py               # Kiểm tra tính tương thích của wrappers
│   └── test_model.py             # Kiểm tra kích thước tensor đầu ra của mạng
├── checkpoints/                  # Thư mục lưu trọng số mô hình (.pth)
├── logs/                         # Thư mục lưu nhật ký Tensorboard (.tfevents)
├── recordings/                   # Thư mục lưu video MP4 kết quả chơi của Mario
├── docs/                         # Báo cáo học thuật, slide thuyết trình, tài liệu hướng dẫn
├── requirements.txt              # Danh sách thư viện Python cần thiết
├── train.py                      # File thực thi huấn luyện chính
├── evaluate.py                   # File thực thi đánh giá định lượng độc lập
├── demo.py                       # File chạy demo trực quan với Pygame/Video recording
└── README.md                     # Tài liệu hướng dẫn cài đặt và chạy chương trình
```

### 6.2. Quy trình Git Flow và Quản lý nhánh (Branching Strategy)
- Nhánh `main`: Chỉ chứa mã nguồn ổn định tuyệt đối, đã được kiểm thử, chỉ cập nhật vào cuối mỗi tuần sau khi họp Sprint Review. Nghiêm cấm push trực tiếp lên `main`.
- Nhánh `develop`: Nhánh tích hợp chung của nhóm. Tất cả các tính năng hoàn thành sẽ được merge vào đây thông qua Pull Request (PR).
- Nhánh tính năng cá nhân (Feature Branches): Đặt tên theo chuẩn:
  - `feature/env-wrappers` (Người 1)
  - `feature/dueling-dqn-agent` (Người 2)
  - `feature/reward-shaping-v2` (Người 3)
  - `feature/eval-and-recording` (Người 4)
- **Quy tắc Pull Request:** Mỗi PR bắt buộc phải có ít nhất 01 thành viên khác review và approve trước khi merge vào `develop`. Người 4 có quyền phê duyệt kỹ thuật cuối cùng để tránh xung đột mã nguồn (Merge Conflicts).
- **Quy ước Commit Message (Conventional Commits):**
  - `feat: add FrameStack wrapper for Gymnasium`
  - `fix: resolve memory leak in replay buffer`
  - `docs: update chapter 3 of report`
  - `test: add unit test for tensor shape in dqn_network`

### 6.3. Quy chế làm việc nhóm và Lịch họp định kỳ
- **Họp Daily Standup (10-15 phút, 3 lần/tuần vào Thứ 2, Thứ 4, Thứ 6 qua Google Meet / Discord):**
  - Hôm qua đã làm được gì?
  - Hôm nay sẽ làm gì?
  - Đang gặp khó khăn, vướng mắc kỹ thuật gì cần sự hỗ trợ của ai?
- **Họp Sprint Review & Planning (60 phút, Chủ Nhật hàng tuần):**
  - Nghiệm thu sản phẩm đầu ra của tuần.
  - Cập nhật bảng phân công công việc và giải quyết rủi ro phát sinh.
  - Chốt kế hoạch và đầu ra bắt buộc cho tuần tiếp theo.

---

## 7. KẾ HOẠCH THỰC HIỆN BÁO CÁO HỌC THUẬT

Báo cáo học thuật cuối kỳ là sản phẩm chiếm trọng số đánh giá rất lớn của giảng viên. Kế hoạch biên soạn được rải đều trong 5 tuần, phân công trách nhiệm rõ ràng cho từng chương:

| Chương | Tiêu đề chương | Nội dung cốt lõi | Người viết chính | Người phản biện/Rà soát | Deadline nộp bản thảo |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Mở đầu** | Lời cảm ơn, Tóm tắt (Abstract), Mục lục | Giới thiệu tóm lược công trình và kết quả đạt được | **Người 4** | Cả nhóm | Tuần 5 (Ngày 31) |
| **Chương 1** | Giới thiệu đề tài & Mục tiêu nghiên cứu | Bối cảnh game NES Mario, bài toán AI chơi game, mục tiêu 8đ và 2đ | **Người 1** | Người 4 | Tuần 2 (Ngày 14) |
| **Chương 2** | Cơ sở lý thuyết về Học tăng cường | RL cơ bản, MDP, Bellman Equation, Q-Learning, DQN, Double DQN, Dueling DQN | **Người 2** | Người 3 | Tuần 2 (Ngày 14) |
| **Chương 3** | Phân tích bài toán Super Mario Bros dưới góc nhìn RL | Không gian quan sát ảnh thô, Frame skip, Frame stack, Action space 7 hành động | **Người 1** | Người 2 | Tuần 2 (Ngày 14) |
| **Chương 4** | Thiết kế Hệ thống và Kiến trúc Mô hình | Sơ đồ khối tổng thể, kiến trúc Nature CNN, luồng Value/Advantage, Replay Buffer | **Người 2** | Người 1 | Tuần 3 (Ngày 21) |
| **Chương 5** | Kỹ thuật Thiết kế Hàm phần thưởng (Reward Shaping) | Phân tích các bẫy hành vi, công thức Reward v1 vs v2, cơ chế phạt chết, thưởng cờ | **Người 3** | Người 2 | Tuần 3 (Ngày 21) |
| **Chương 6** | Thiết lập Huấn luyện & Quy trình Thực nghiệm | Siêu tham số (Hyperparameters), tài nguyên phần cứng (GPU/CPU), TensorBoard tracking | **Người 3** | Người 4 | Tuần 3 (Ngày 21) |
| **Chương 7** | Kết quả Thực nghiệm Giải màn World 1-1 | Bảng đánh giá 20 seeds độc lập, Win Rate, Max X-Pos, phân tích các pha nhảy mẫu | **Người 3** | Người 4 | Tuần 4 (Ngày 28) |
| **Chương 8** | Các Nghiên cứu Mở rộng & Ablation Study | Đối sánh Dueling DDQN vs PPO, Ablation Reward Shaping, Không gian hành động | **Người 2, 3** | Người 4 | Tuần 4 (Ngày 28) |
| **Chương 9** | Hạn chế của Hệ thống và Hướng phát triển | Hiện tượng quá khớp World 1-1, khả năng tổng quát hóa sang World 1-2, hạn chế RAM | **Người 1** | Người 3 | Tuần 4 (Ngày 28) |
| **Chương 10** | Hướng dẫn Cài đặt, Sử dụng & Kiểm thử Tái lập | Hướng dẫn chi tiết từng câu lệnh từ môi trường trắng đến chạy demo và train | **Người 4** | Người 1 | Tuần 5 (Ngày 30) |
| **Kết luận** | Tổng kết đề tài & Tài liệu tham khảo | Tóm tắt đóng góp kỹ thuật, danh mục 15-20 bài báo khoa học chuẩn trích dẫn IEEE | **Người 4** | Cả nhóm | Tuần 5 (Ngày 31) |

---

## 8. KẾ HOẠCH XÂY DỰNG SLIDE VÀ DIỄN TẬP THUYẾT TRÌNH

### 8.1. Cấu trúc Slide thuyết trình chuẩn mực (Dự kiến 25-28 Slides, 20 phút trình bày)
- **Slide 1:** Trang bìa (Tên đề tài, Giảng viên hướng dẫn, Danh sách 4 thành viên và vai trò).
- **Slide 2:** Đặt vấn đề và Thách thức trong bài toán Mario (Tại sao Mario khó với RL?).
- **Slide 3:** Mục tiêu đề tài (Mục tiêu cốt lõi 8 điểm & Nghiên cứu mở rộng 2 điểm).
- **Slide 4-6 (Người 1 trình bày - 5 phút):**
  - Mô hình hóa bài toán MDP.
  - Kiến trúc Pipeline tiền xử lý: Giải thích tại sao cần Grayscale, Resize $84 \times 84$, FrameSkip 4 và FrameStack 4.
  - Tối ưu hóa không gian hành động rời rạc (`SIMPLE_MOVEMENT`).
- **Slide 7-11 (Người 2 trình bày - 5 phút):**
  - Tại sao chọn Dueling Double DQN thay vì Standard DQN?
  - Cơ chế toán học tách biệt Target Network và Online Network giảm Overestimation bias.
  - Kiến trúc mạng Dueling CNN: Phân tách luồng Giá trị trạng thái $V(s)$ và Lợi thế hành động $A(s, a)$.
  - Sơ đồ luồng dữ liệu Replay Buffer và cập nhật gradient.
- **Slide 12-16 (Người 3 trình bày - 5 phút):**
  - Thiết kế Reward Shaping: Cách giải bài toán Mario sợ chết, kẹt ống nước, dậm chân tại chỗ.
  - Quá trình huấn luyện thực tế và đường cong hội tụ trên TensorBoard qua các mốc 500k, 1M, 2M steps.
  - Các nghiên cứu triệt tiêu (Ablation Studies): So sánh Dueling DDQN vs PPO, Shaped Reward vs Sparse Reward.
- **Slide 17-22 (Người 4 trình bày - 5 phút):**
  - Kết quả kiểm thử định lượng độc lập trên 20 seeds ngẫu nhiên (Tỷ lệ thắng, X-pos trung bình).
  - Trình chiếu Video Demo trực tiếp Mario vượt màn World 1-1 với HUD trực quan.
  - Kiến trúc mã nguồn, tính tái lập (Reproducibility) và hướng dẫn chạy kiểm thử.
  - Tổng kết, bài học kinh nghiệm và định hướng mở rộng sang các World tiếp theo.
- **Slide 23-25:** Q&A và Lời cảm ơn.

### 8.2. Kế hoạch Diễn tập Thuyết trình và Vấn đáp thử (Mock Defense)
- **Buổi 1 (Ngày 34):** Tập trung vào việc khớp thời gian (Time Management). Đảm bảo mỗi thành viên trình bày đúng trong khoảng 4-5 phút, tổng thời gian không vượt quá 20 phút quy định.
- **Buổi 2 (Ngày 35):** Tập trung vào phong thái, slide chuyển tiếp mượt mà và trả lời phản biện bất ngờ. Mỗi thành viên chuẩn bị sẵn tâm thế trả lời câu hỏi thuộc chuyên môn sâu của mình và câu hỏi kiến thức nền tảng của toàn hệ thống.

---

## 9. BỘ CÂU HỎI VÀ ĐÁP ÁN ÔN TẬP VẤN ĐÁP CHUYÊN SÂU

Bộ câu hỏi này được thiết kế dựa trên các câu hỏi thường gặp nhất của Hội đồng chấm đồ án Học tăng cường:

### Câu 1: Tại sao nhóm phải dùng Frame Stacking (ghép 4 khung hình) thay vì chỉ đưa 1 khung hình xám vào mạng nơ-ron?
- **Đáp án chuẩn:** Trò chơi Super Mario Bros là một môi trường quan sát một phần nếu chỉ nhìn 1 khung hình tĩnh (Partially Observable Markov Decision Process - POMDP). Từ một bức ảnh duy nhất, tác nhân không thể biết được Mario đang đứng yên, chạy sang phải hay nhảy sang trái; cũng không thể biết được con Goomba đang di chuyển theo hướng nào hay đạn pháo đang bay với tốc độ bao nhiêu. Bằng cách ghép 4 khung hình liên tiếp $(4, 84, 84)$, mạng CNN có thể trích xuất được thông tin về **hướng chuyển động, vận tốc và gia tốc** của mọi vật thể trong game, đưa bài toán trở về đúng tính chất Markov đầy đủ (Fully Observable MDP).

### Câu 2: Hiện tượng Overestimation Bias trong Standard DQN là gì và Double DQN giải quyết nó như thế nào?
- **Đáp án chuẩn:** Trong Standard DQN, mục tiêu cập nhật được tính bằng $Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-)$. Toán tử $\max$ đồng thời dùng chính bộ trọng số của mạng đích để vừa **lựa chọn** hành động có Q-value lớn nhất vừa **đánh giá** giá trị đó. Do hàm xấp xỉ mạng nơ-ron luôn có sai số ngẫu nhiên, việc lấy $\max$ trên các giá trị có sai số sẽ làm giá trị Q kỳ vọng bị thổi phồng một cách có hệ thống (Overestimation Bias), dẫn đến chính sách chệch hướng. Double DQN giải quyết vấn đề này bằng cách tách rời hai tác vụ: Mạng Online ($\theta$) dùng để **chọn hành động tốt nhất** $a^* = \arg\max_a Q(S_{t+1}, a; \theta)$, còn Mạng Target ($\theta^-$) dùng để **đánh giá giá trị của hành động đó**: $Y_t = R_{t+1} + \gamma Q(S_{t+1}, a^*; \theta^-)$.

### Câu 3: Kiến trúc Dueling DQN có ưu thế gì vượt trội so với DQN truyền thống trong game Mario?
- **Đáp án chuẩn:** Dueling DQN chia tầng Fully Connected thành hai nhánh riêng biệt: một nhánh ước lượng Giá trị trạng thái $V(s)$ (đo lường mức độ tốt của việc đang đứng ở vị trí $s$) và một nhánh ước lượng Ưu thế hành động $A(s, a)$ (mức độ vượt trội của hành động $a$ so với các hành động khác). Trong game Mario, có rất nhiều trạng thái mà mọi hành động đều mang lại kết quả như nhau (ví dụ: Mario đang rơi tự do trên không trung hoặc đứng ở vùng đất hoàn toàn bằng phẳng không có quái vật, việc bấm nút hay không bấm nút không làm thay đổi giá trị của trạng thái). Dueling DQN cho phép mạng học giá trị $V(s)$ một cách trực tiếp và chính xác mà không cần phải trải nghiệm toàn bộ các hành động tại trạng thái đó, giúp tốc độ hội tụ nhanh hơn đáng kể.

### Câu 4: Giải thích hiện tượng tác nhân Mario bị "kẹt ở ống nước" hoặc "nhảy lùi/đứng yên" trong quá trình huấn luyện và nhóm đã khắc phục bằng cách nào?
- **Đáp án chuẩn:** Đây là hiện tượng tác nhân rơi vào cực tiểu cục bộ (Local Minima). Ống nước cao đòi hỏi Mario phải chạy lấy đà và nhấn giữ phím nhảy trong thời gian đủ dài. Khi chưa học được kỹ năng này, việc đâm vào ống nước hoặc đi tiếp sẽ đối mặt với nguy cơ bị Goomba giết chết (bị phạt nặng $-50$). Tác nhân phát hiện ra rằng việc đứng yên một chỗ hoặc đi lùi về bên trái sẽ an toàn hơn và không bị phạt chết. Nhóm đã khắc phục bằng 3 kỹ thuật:
  1. *Anti-stuck Penalty:* Phạt điểm âm nếu tọa độ X của Mario không tăng thêm quá 5 pixel trong 50 bước liên tiếp.
  2. *Thưởng gia tốc hướng tới:* Bổ sung trọng số thưởng cho việc duy trì vận tốc dương theo trục X.
  3. *Tối ưu hóa không gian hành động:* Chuyển sang dùng `RIGHT_ONLY` hoặc `SIMPLE_MOVEMENT` với tổ hợp phím `Right + A` và `Right + A + B` được ưu tiên trong chính sách khởi tạo.

### Câu 5: Tiêu chí nào chứng minh một cách khoa học rằng tác nhân thực sự giải được màn World 1-1 chứ không phải ăn may?
- **Đáp án chuẩn:**
  1. *Điều kiện chiến thắng khách quan từ môi trường:* Mario phải kích hoạt cờ chiến thắng với biến trạng thái `info['flag_get'] == True` được trả về từ engine của game NES Mario, tương đương tọa độ $X \ge 3161$.
  2. *Đánh giá định lượng trên tập kiểm thử độc lập:* Tác nhân được tắt hoàn toàn tính ngẫu nhiên ($\epsilon = 0.0$) và chạy trên 20 seeds ngẫu nhiên khác nhau. Tỷ lệ hoàn thành màn chơi (Win Rate) phải đạt $\ge 80\%$.
  3. *Độ ổn định của số bước:* Tác nhân hoàn thành màn chơi trong khoảng thời gian tối ưu ($200-300$ steps ở môi trường bỏ qua frame, tương đương với người chơi thực thụ).

---

## 10. TIÊU CHÍ KIỂM TRA VÀ NGHIỆM THU SẢN PHẨM

| Hạng mục nghiệm thu | Chỉ số kỹ thuật bắt buộc | Phương pháp kiểm tra | Người chịu trách nhiệm | Trạng thái |
| :--- | :--- | :--- | :---: | :---: |
| **1. Môi trường Gym** | Chạy 1,000 steps liên tục không crash, không rò rỉ RAM ($< 1.5$ GB) | Chạy script `tests/test_env.py` | Người 1 | [ ] Đạt |
| **2. Tương thích Tensor** | Input shape `(Batch, 4, 84, 84)`, Output shape `(Batch, 7)` | Chạy Pytest `tests/test_model.py` | Người 2 | [ ] Đạt |
| **3. Pipeline Huấn luyện** | Tốc độ huấn luyện $\ge 80$ FPS (GPU) hoặc $\ge 30$ FPS (CPU) | Đo FPS thực tế trong `train.py` | Người 2, 3 | [ ] Đạt |
| **4. Lưu & Tải Checkpoint** | Nạp file `.pth` và tiếp tục huấn luyện/đánh giá mà không sai lệch trọng số | Chạy script nạp checkpoint thử nghiệm | Người 2 | [ ] Đạt |
| **5. Mục tiêu chính (8 điểm)** | Mario chạm cờ `flag_get == True`, Win Rate $\ge 80\%$ trên 20 test episodes | Chạy `evaluate.py --episodes 20` | Cả nhóm | [ ] Đạt |
| **6. Cải tiến mở rộng (2 điểm)** | Có dữ liệu đối sánh định lượng rõ ràng giữa 2 thuật toán hoặc 2 hàm reward | Kiểm tra đồ thị và bảng số liệu trên TensorBoard | Người 3, 2 | [ ] Đạt |
| **7. Tính tái lập (Reproducibility)** | Cài đặt và chạy thành công trên máy tính thứ hai độc lập chỉ qua README | Người 4 clone repo và chạy thử nghiệm sạch | Người 4 | [ ] Đạt |
| **8. Video Demo** | Video định dạng MP4 full-HD, có âm thanh hoặc HUD hiển thị thông số AI | Mở kiểm tra file trong `recordings/` | Người 4 | [ ] Đạt |
| **9. Báo cáo học thuật** | Đầy đủ 10-12 chương, chuẩn định dạng Times New Roman, trích dẫn IEEE | Rà soát file PDF hoàn chỉnh | Người 4 (Lead) | [ ] Đạt |
| **10. Diễn tập vấn đáp** | Cả 4 thành viên trả lời đúng $\ge 90\%$ các câu hỏi trong bộ ngân hàng đề | Thực hiện 2 buổi Mock Defense | Cả nhóm | [ ] Đạt |

---

## 11. BẢNG QUẢN LÝ RỦI RO VÀ KẾ HOẠCH ỨNG PHÓ

| Mã | Rủi ro tiềm ẩn | Xác suất | Tác động | Dấu hiệu nhận biết | Biện pháp phòng ngừa | Kế hoạch xử lý khi xảy ra sự cố (Contingency Plan) |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **R01** | Lỗi cài đặt `nes-py` trên Python 3.12/3.13 (thiếu MSVC C++) | **Cao** | **Nghiêm trọng** | `pip install` báo lỗi biên dịch C++ hoặc wheel not found | Thống nhất sử dụng Python 3.10 hoặc 3.11 trong môi trường Conda/venv chuẩn | Người 1 cung cấp file Dockerfile hoặc pre-built wheel / shimmy wrapper cho cả nhóm |
| **R02** | Xung đột API Gymnasium (`reset()`, `step()`) với Gym cũ | **Cao** | **Nghiêm trọng** | `ValueError: too many values to unpack` (5 values vs 4 values) | Viết sẵn Custom Wrapper chuyển đổi chuẩn Gymnasium ngay từ Tuần 1 | Sử dụng class wrapper chuẩn hóa `step()` trả về đúng `(obs, reward, terminated, truncated, info)` |
| **R03** | Tác nhân rơi vào bẫy cục bộ (kẹt ở ống nước, sợ chết nhảy lùi) | **Cao** | **Nghiêm trọng** | Đường cong Reward đi ngang sau 500k steps; Mario không qua được x=500 | Phân tích log TensorBoard hàng ngày; thiết kế Reward có tính định hướng tiến tới | Kích hoạt Reward v2: phạt đứng yên (Anti-stuck), tăng epsilon ngẫu nhiên tạm thời |
| **R04** | Thời gian huấn luyện quá lâu, không đủ tài nguyên GPU cá nhân | **Trung bình** | **Nghiêm trọng** | Tốc độ học $< 20$ FPS; dự kiến mất hơn 5 ngày để chạy 2 triệu steps | Tối ưu hóa FrameSkip = 4, kích thước ảnh $84 \times 84$; Replay buffer dùng RAM hiệu quả | Chuyển toàn bộ pipeline lên Google Colab Pro / Kaggle GPU (T4/P100) chạy nền song song |
| **R05** | Hỏng ổ cứng hoặc mất checkpoint mô hình do cúp điện | **Thấp** | **Nghiêm trọng** | Không tìm thấy file trọng số `.pth` khi cần đánh giá | Thiết lập script tự động lưu checkpoint mỗi 100k steps và lưu top 3 checkpoint tốt nhất | Người 3 tự động đẩy checkpoint tốt lên Google Drive / Hugging Face Model Hub sau mỗi đợt chạy |
| **R06** | Xung đột mã nguồn (Merge Conflict) làm hỏng nhánh `main` | **Trung bình** | **Vừa** | Không thể merge PR trên GitHub; code bị đè | Nghiêm cấm push trực tiếp vào `main`; áp dụng quy tắc 1 PR chỉ giải quyết 1 tính năng | Người 4 chủ trì phiên gỡ conflict thủ công qua Git GUI hoặc VS Code |
| **R07** | Một thành viên bị ốm hoặc chậm tiến độ nghiêm trọng | **Trung bình** | **Vừa** | Công việc quá hạn 3 ngày; vắng mặt trong 2 buổi Standup liên tiếp | Bảng phân công luôn có Người phối hợp (Backup Owner) nắm rõ nghiệp vụ | Kích hoạt cơ chế hỗ trợ chéo: Người 4 và Người phối hợp tiếp quản nhiệm vụ đó ngay |
| **R08** | Không kịp thời gian cho phần mở rộng 2 điểm cộng | **Trung bình** | **Thấp** | Đến cuối tuần 3 vẫn chưa đạt mốc World 1-1 để chuyển sang mở rộng | Tuyệt đối tuân thủ nguyên tắc: Đạt trọn vẹn 8 điểm P0 mới phân bổ tài nguyên cho P2 | Nếu chậm, chỉ thực hiện 1 nghiên cứu triệt tiêu đơn giản nhất (Ablation Reward) thay vì đổi thuật toán |

---

## 12. CHECKLIST HOÀN THÀNH DỰ ÁN TOÀN DIỆN

Trước khi nộp bài cho Giảng viên, Trưởng nhóm và các thành viên phải tự kiểm tra và đánh dấu tích vào từng tiêu chí sau:

- [ ] **Mã nguồn (Source Code):**
  - [ ] Thư mục mã nguồn sạch sẽ, không chứa file tạm, cache `__pycache__`, checkpoint rác.
  - [ ] File `requirements.txt` phản ánh chính xác các thư viện cần thiết.
  - [ ] File `README.md` hướng dẫn chi tiết cách cài đặt và chạy trong 3 bước.
  - [ ] Script `evaluate.py` chạy độc lập và cho ra kết quả trùng khớp với báo cáo.
  - [ ] Script `demo.py` mở được cửa sổ chơi trực quan của Mario.
- [ ] **Mô hình huấn luyện (Weights):**
  - [ ] File trọng số `checkpoints/dueling_ddqn_world1_1_best.pth` được lưu trữ an toàn và nạp thành công.
  - [ ] Có kèm link tải dự phòng từ Google Drive nếu file vượt dung lượng GitHub.
- [ ] **Báo cáo học thuật (Report):**
  - [ ] Định dạng chuẩn A4, Font Times New Roman, đầy đủ Mục lục, Danh mục hình vẽ, Bảng biểu.
  - [ ] Đủ 10-12 chương với nội dung kỹ thuật sâu sắc, công thức toán học KaTeX rõ ràng.
  - [ ] Có trích dẫn tối thiểu 10 bài báo khoa học uy tín (Mnih et al., Van Hasselt et al., Wang et al., Schulman et al.).
- [ ] **Slide thuyết trình & Demo:**
  - [ ] Slide thiết kế thẩm mỹ cao, ít chữ, nhiều sơ đồ trực quan và biểu đồ TensorBoard.
  - [ ] Video demo Mario chạm cờ World 1-1 có sẵn ở định dạng MP4 offline trong máy tính.
  - [ ] Phân chia thời lượng nói đồng đều cho cả 4 thành viên (mỗi người 5 phút).
- [ ] **Kỹ năng vấn đáp (Defense Readiness):**
  - [ ] Cả 4 thành viên đã thuộc lòng câu trả lời cho 5 câu hỏi trọng tâm của đề tài.
  - [ ] Đã hoàn thành 02 buổi diễn tập bảo vệ thử nghiệm có bấm giờ.
