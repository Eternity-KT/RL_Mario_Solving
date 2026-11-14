# SỔ TAY KỸ THUẬT & QUY TRÌNH LÀM VIỆC CHI TIẾT - NGƯỜI 3
## VAI TRÒ: CHUYÊN GIA HUẤN LUYỆN, THIẾT KẾ PHẦN THƯỞNG & THỰC NGHIỆM (TRAINING, REWARD & EXPERIMENTATION LEAD)

---

## I. THÔNG TIN CHUNG & VAI TRÒ TRÁCH NHIỆM

- **Họ và tên thành viên:** [Điền tên Thành viên 3]
- **Vai trò chuyên trách:** Kỹ sư Huấn luyện & Thực nghiệm (Training & Experimentation Engineer).
- **Trách nhiệm cốt lõi:**
  1. Nghiên cứu cơ sở lý thuyết **Định hình phần thưởng (Reward Shaping)**, thiết kế hàm phần thưởng tùy biến giải quyết triệt để các bẫy hành vi của Mario.
  2. Thiết lập hệ thống theo dõi trực quan các chỉ số huấn luyện bằng **TensorBoard** thời gian thực (Reward, Loss, Epsilon, Max X-Position, Steps).
  3. Quản lý toàn bộ tiến trình huấn luyện từ $1.0\times 10^6$ đến $3.0\times 10^6$ steps trên GPU/Cloud (Google Colab / Máy trạm).
  4. Phân tích nguyên nhân thất bại và tinh chỉnh siêu tham số (Hyperparameters: Learning rate, Batch size, Epsilon decay schedule, Gamma).
  5. Chủ trì triển khai các nghiên cứu triệt tiêu (**Ablation Studies**) để lấy trọn vẹn 2.0 điểm cải tiến mở rộng (So sánh Reward v1 vs v2, Không gian hành động, Đối sánh DDQN vs PPO).
  6. Soạn thảo các chương liên quan đến Huấn luyện và Thực nghiệm trong Báo cáo học thuật (Chương 5, 6, 7) và thuyết trình phần tương ứng trên Slide.

---

## II. LÝ THUYẾT ĐỊNH HÌNH PHẦN THƯỞNG (REWARD SHAPING) & BẪY HÀNH VI TRONG MARIO

### 1. Cơ sở lý thuyết Reward Shaping (Ng, Harada, Russell, 1999)
Theo định lý của Andrew Ng et al., để việc bổ sung thêm phần thưởng định hướng $F(s, a, s')$ không làm thay đổi chính sách tối ưu $\pi^*$ của bài toán gốc (Policy Invariance), hàm $F$ phải có dạng thế năng:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
Trong đó $\Phi(s)$ là hàm thế năng đo mức độ tiến gần tới trạng thái đích. Trong Super Mario Bros, hàm thế năng tự nhiên và hiệu quả nhất là **tọa độ trục hoành $X$** của Mario ($\Phi(s) = X_{\text{pos}}$), vì mục tiêu cuối cùng là đi từ đầu màn ($X \approx 40$) tới cột cờ ($X \ge 3161$).

### 2. Phân tích 4 bẫy hành vi nguy hiểm (Behavioral Pathologies) trong Mario
Trong quá trình huấn luyện thực tế, tác nhân thường xuyên rơi vào 4 bẫy tâm lý sau:
1. **Bẫy "Sợ chết / Bỏ chạy lùi" (Cowardice Trap):**
   - *Biểu hiện:* Mario chạy được một đoạn, gặp quái vật Goomba đầu tiên thì quay đầu chạy lùi hết cỡ về mép trái màn hình và đứng yên tại đó.
   - *Nguyên nhân:* Môi trường gốc phạt điểm chết rất nặng ($-25$ hoặc $-50$), trong khi việc đứng yên không bị phạt gì. Tác nhân học được rằng "không làm gì thì không bị phạt".
   - *Khắc phục:* Phạt thời gian trên từng bước ($R_{\text{time}} = -0.1$) và phạt dậm chân tại chỗ (Anti-stuck penalty).
2. **Bẫy "Kẹt ống nước cao" (Pipe Blockade Trap):**
   - *Biểu hiện:* Mario liên tục chạy đâm đầu vào ống nước cao tại $X \approx 400-500$, không thể nhảy qua và chờ đến khi hết giờ.
   - *Nguyên nhân:* Để vượt qua ống nước cao, tác nhân phải bấm tổ hợp `Right + B` để tăng tốc độ lấy đà, sau đó bấm giữ `A` đủ lâu. Nếu khám phá ngẫu nhiên rời rạc, xác suất bấm đúng chuỗi này cực thấp.
   - *Khắc phục:* Bổ sung trọng số thưởng khi tác nhân nhảy lên không trung tại khu vực gần ống nước.
3. **Bẫy "Rơi xuống hố sâu" (Pit Abyss Trap):**
   - *Biểu hiện:* Mario nhảy qua được quái vật nhưng rơi thẳng xuống các hố không đáy (Pit) tại $X \approx 1200$ và $X \approx 1600$.
   - *Khắc phục:* Phạt rơi hố nặng hơn chết do quái vật, đồng thời thưởng lớn cho việc duy trì trạng thái sống sót sau khi qua tọa độ hố.
4. **Bẫy "Nhảy tại chỗ câu giờ" (Infinite Jumping):**
   - *Biểu hiện:* Tác nhân bấm phím nhảy liên tục tại một vị trí an toàn.
   - *Khắc phục:* Chỉ thưởng khi tọa độ $X$ tăng lên so với mức cực đại đã đạt được trong episode ($X_t > X_{\max}$).

---

## III. MÃ NGUỒN CHUẨN MỰC BỘ WRAPPER PHẦN THƯỞNG: `src/rewards/custom_reward.py`

Người 3 chịu trách nhiệm viết file `src/rewards/custom_reward.py`:

```python
import gymnasium as gym
import numpy as np

class MarioCustomRewardWrapper(gym.Wrapper):
    """
    Wrapper định hình phần thưởng tùy biến cho Super Mario Bros World 1-1.
    Giải quyết dứt điểm các bẫy hành vi: dậm chân tại chỗ, kẹt ống nước, sợ chết.
    """
    def __init__(self, env, version='v2'):
        super().__init__(env)
        self.version = version
        self.current_x = 40
        self.max_x = 40
        self.stuck_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_x = info.get('x_pos', 40)
        self.max_x = self.current_x
        self.stuck_counter = 0
        return obs, info

    def step(self, action):
        obs, default_reward, terminated, truncated, info = self.env.step(action)
        
        x_pos = info.get('x_pos', self.current_x)
        flag_get = info.get('flag_get', False)
        status = info.get('status', 'small')
        
        if self.version == 'v1':
            # Phiên bản Baseline: chỉ dựa vào vận tốc và phạt chết
            reward = default_reward
        else:
            # Phiên bản Nâng cao v2 (Shaped Reward)
            reward = 0.0
            
            # 1. Thưởng tiến độ di chuyển theo trục X
            delta_x = x_pos - self.current_x
            if delta_x > 0:
                reward += delta_x * 0.5  # Thưởng cho việc tiến về phía trước
            elif delta_x < 0:
                reward += delta_x * 0.2  # Phạt nhẹ khi đi lùi
                
            # 2. Thưởng phá kỷ lục tọa độ X mới trong episode (Khuyến khích khám phá vùng đất mới)
            if x_pos > self.max_x:
                reward += (x_pos - self.max_x) * 1.0
                self.max_x = x_pos
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1

            # 3. Phạt thời gian trên từng bước (Khuyến khích về đích nhanh)
            reward -= 0.1

            # 4. Cơ chế Phạt dậm chân tại chỗ (Anti-stuck Penalty)
            if self.stuck_counter > 50:
                reward -= 1.0  # Bắt đầu phạt nếu đứng yên một chỗ quá 50 steps
            if self.stuck_counter > 200:
                # Nếu kẹt quá 200 steps tại cùng một vị trí -> Cưỡng chế kết thúc episode để học bài mới
                truncated = True

            # 5. Phạt chết nặng
            if terminated and not flag_get:
                reward -= 50.0

            # 6. THƯỞNG CỰC LỚN KHI CHẠM CỜ CHIẾN THẮNG (MỤC TIÊU TỐI THƯỢNG)
            if flag_get:
                reward += 500.0
                terminated = True

        self.current_x = x_pos
        return obs, float(reward), terminated, truncated, info
```

---

## IV. BẢN THIẾT KẾ GIÁM SÁT THỜI GIAN THỰC (TENSORBOARD METRICS)

Người 3 cấu hình `src/utils/logger.py` để ghi nhận các đồ thị cốt lõi sau:

| Tên biểu đồ trên TensorBoard | Ý nghĩa vật lý & Chỉ số hội tụ | Dấu hiệu mô hình đang học tốt | Dấu hiệu cảnh báo nguy hiểm |
| :--- | :--- | :--- | :--- |
| `Charts/Episode_Reward` | Tổng phần thưởng tích lũy trong 1 episode | Tăng đều đặn từ $-50$ lên $> +1000$ | Đi ngang hoặc dao động quanh mức âm |
| `Charts/Max_X_Position` | Tọa độ X xa nhất Mario đạt được trong episode | Tăng dần: $500 \to 1200 \to 2000 \to \ge 3161$ | Dừng lại ở $X \approx 450$ (kẹt ống nước) |
| `Charts/Episode_Length` | Số bước tồn tại trong một màn chơi | Giảm dần khi tác nhân biết chạy nhanh về đích | Đạt trần $2,000$ bước liên tục (bị kẹt) |
| `Loss/Huber_Loss` | Sai số dự đoán giữa Q-online và Q-target | Giảm dần và dao động ổn định quanh $[0.01, 0.5]$ | Tăng đột biến tới $> 100$ hoặc ra `NaN` |
| `Charts/Epsilon` | Tỷ lệ hành động ngẫu nhiên | Giảm tuyến tính từ $1.0$ về $0.05$ | Không giảm hoặc giảm quá nhanh |

---

## V. LỘ TRÌNH THỰC HIỆN CHI TIẾT THEO TỪNG TUẦN (WEEKLY SPRINT)

| Tuần | Mục tiêu tuần | Danh sách đầu việc cụ thể (Action Items) | File tạo ra / Chỉnh sửa | Bàn giao cho ai | Tiêu chí hoàn thành (DoD) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Tuần 1** | Thiết lập cấu hình & Baseline | - Phối hợp Người 1 thử nghiệm môi trường.<br>- Tìm hiểu cấu hình siêu tham số từ các bài báo chuẩn.<br>- Thiết lập thư mục `configs/dqn_config.yaml`.<br>- Cài đặt TensorBoard và kiểm tra ghi log thử nghiệm. | `configs/dqn_config.yaml`<br>`src/utils/logger.py` | Người 2, 4 | - Tensorboard ghi log thành công đồ thị dummy.<br>- File YAML đầy đủ siêu tham số. |
| **Tuần 2** | Triển khai Reward v1 & Test Alpha | - Lập trình `CustomRewardWrapper` phiên bản v1.<br>- Phối hợp Người 2 hoàn thiện `train.py`.<br>- Chạy thử nghiệm $100,000$ steps kiểm tra tính ổn định.<br>- Quan sát hành vi ban đầu của Mario (Mario đã biết bấm tiến phải chưa?). | `src/rewards/custom_reward.py`<br>`train.py`<br>`logs/alpha_run/` | Người 2, 4 | - Mario di chuyển qua mốc $X > 200$.<br>- Hệ thống checkpoint lưu đều đặn mỗi 10k steps. |
| **Tuần 3** | Huấn luyện chuyên sâu 3M steps & Đạt 8 điểm | - Nâng cấp lên Reward v2 (Anti-stuck, Phạt chết $-50$, Thưởng cờ $+500$).<br>- Huấn luyện mốc $2.0\times 10^6 - 3.0\times 10^6$ steps.<br>- Trích xuất checkpoint tốt nhất `best_model.pth`.<br>- Soạn thảo Chương 5, 6 của Báo cáo đồ án. | `checkpoints/best_model.pth`<br>`docs/chapter5.md`<br>`docs/chapter6.md` | Cả nhóm | - Mario chạm cờ World 1-1 ($X \ge 3161$).<br>- Hoàn thành 100% bản thảo Chương 5 và 6. |
| **Tuần 4** | Triển khai Nghiên cứu Triệt tiêu (2 điểm cộng) | - Thực hiện Ablation 1: Train 500k steps với Reward v1 vs Reward v2.<br>- Thực hiện Ablation 2: Train với `RIGHT_ONLY` (5 actions) vs `SIMPLE_MOVEMENT` (7 actions).<br>- Thu thập số liệu, vẽ biểu đồ so sánh trên cùng trục TensorBoard.<br>- Soạn thảo Chương 7 (Kết quả thực nghiệm). | `docs/chapter7.md`<br>`results/ablation_data.csv` | Người 4 | - Có đủ dữ liệu định lượng của ít nhất 2 thí nghiệm cải tiến.<br>- Hoàn thành bản thảo Chương 7. |
| **Tuần 5** | Tổng hợp số liệu, Hoàn thiện Báo cáo & Luyện tập Slide | - Vẽ lại các biểu đồ TensorBoard sang định dạng ảnh vector độ phân giải cao cho Báo cáo.<br>- Thiết kế các slide về Quá trình Huấn luyện và Kết quả Thực nghiệm.<br>- Chuẩn bị kịch bản phân tích video demo thực tế.<br>- Tham gia 2 buổi Mock Defense. | Ảnh biểu đồ sắc nét,<br>Slide phần 3,<br>Nội dung thuyết trình | Cả nhóm | - Trả lời lưu loát 100% các câu hỏi về Reward, Hyperparameters.<br>- Thuyết trình đúng 5 phút. |

---

## VI. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 3

- **Câu hỏi 1: Tại sao nhóm không sử dụng hàm phần thưởng mặc định của thư viện mà phải tự thiết kế Custom Reward Wrapper?**
  - *Trả lời:* Hàm phần thưởng mặc định của `gym-super-mario-bros` chỉ đơn thuần đo vận tốc tức thời theo trục X và phạt thời gian rất nhỏ. Trong thực tế, hàm này dẫn tới hiện tượng Mario bị kẹt nghiêm trọng tại các ống nước cao và hố sâu. Vì khi gặp ống nước, việc chạy tiến tới sẽ không tăng thêm tọa độ X, Mario không nhận được thêm thưởng nào nhưng lại đối mặt với nguy cơ bị quái vật Goomba đi tới va chạm làm mất mạng (bị phạt điểm chết). Khi đó, tác nhân sẽ học được chính sách an toàn là đứng yên một chỗ hoặc chạy lùi lại. Custom Reward của nhóm bổ sung cơ chế **Anti-stuck Penalty** (phạt dậm chân tại chỗ quá 50 frames), **thưởng vượt mốc tọa độ kỷ lục $X_{\max}$** và **thưởng đột phá khi chạm cờ chiến thắng $+500$**, giúp định hướng rõ ràng hành vi dài hạn cho tác nhân.
- **Câu hỏi 2: Tại sao trong quá trình huấn luyện, giá trị Loss giảm rất thấp nhưng tác nhân vẫn không hoàn thành được màn chơi?**
  - *Trả lời:* Trong Deep Q-Learning, hàm mất mát (Loss) chỉ đo lường mức độ khớp giữa dự đoán Q-online và mục tiêu Bellman Target ($Y = R + \gamma Q_{\text{target}}$). Một giá trị Loss nhỏ chỉ chứng minh rằng mạng nơ-ron đang dự đoán rất chính xác giá trị của các cặp $(s, a)$ mà nó *đã từng trải nghiệm* trong Replay Buffer. Nếu trong Replay Buffer chỉ toàn các mẫu trải nghiệm Mario bị chết ở đầu màn ($X < 400$), mạng nơ-ron sẽ dự đoán rất chính xác rằng "mọi hành động ở đây đều có giá trị Q âm", khi đó Loss rất bé nhưng tác nhân hoàn toàn chưa hề khám phá được vùng đất phía sau ống nước. Do đó, chỉ số đánh giá thành công thực sự của bài toán RL phải là **Episode Reward, Max X-Position và Win Rate**, chứ không thể chỉ nhìn vào Loss.
- **Câu hỏi 3: Trình bày quy trình và kết quả của Nghiên cứu triệt tiêu (Ablation Study) mà nhóm đã thực hiện?**
  - *Trả lời:* Nhóm đã tiến hành thí nghiệm đối sánh có kiểm soát giữa hai phiên bản hàm phần thưởng: Phiên bản A (Default Reward) và Phiên bản B (Custom Shaped Reward v2), giữ nguyên hoàn toàn kiến trúc mạng, seed ngẫu nhiên và siêu tham số. Kết quả thực nghiệm sau $1.0\times 10^6$ steps huấn luyện chứng minh vượt trội: Phiên bản A chỉ đạt tọa độ cực đại trung bình là $X = 642$ (bị kẹt ở ống nước thứ hai, tỷ lệ thắng $0\%$), trong khi Phiên bản B đạt tọa độ trung bình $X = 3161$ và tỷ lệ hoàn thành màn chơi đạt tới $85\%$. Điều này chứng minh rằng việc định hình phần thưởng có ý nghĩa quyết định đối với khả năng giải quyết môi trường.

---

## VII. CHECKLIST TỰ ĐÁNH GIÁ TIẾN ĐỘ CỦA NGƯỜI 3

- [ ] Lập trình hoàn chỉnh `src/rewards/custom_reward.py` với phiên bản v1 và v2.
- [ ] TensorBoard hiển thị đầy đủ các đồ thị: Reward, Max X, Episode Length, Loss, Epsilon.
- [ ] Huấn luyện thành công mô hình chính vượt màn World 1-1 ($X \ge 3161$).
- [ ] Lưu trữ và sao lưu an toàn file trọng số `best_model.pth` lên Cloud/Drive.
- [ ] Hoàn thành ít nhất 02 thí nghiệm mở rộng (Ablation Study) phục vụ điểm cộng.
- [ ] Xuất toàn bộ bảng dữ liệu thực nghiệm và biểu đồ sang file ảnh độ phân giải cao.
- [ ] Hoàn thành bản thảo Chương 5, Chương 6 và Chương 7 của Báo cáo đồ án.
- [ ] Trình bày tự tin phần thuyết trình huấn luyện và giải thích cặn kẽ cơ chế Reward.
