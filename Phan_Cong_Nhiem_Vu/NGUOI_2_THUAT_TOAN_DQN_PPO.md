# SỔ TAY KỸ THUẬT & QUY TRÌNH LÀM VIỆC CHI TIẾT - NGƯỜI 2
## VAI TRÒ: CHUYÊN GIA THUẬT TOÁN HỌC TĂNG CƯỜNG & KIẾN TRÚC MẠNG NƠ-RON (ALGORITHM & DEEP LEARNING LEAD)

---

## I. THÔNG TIN CHUNG & VAI TRÒ TRÁCH NHIỆM

- **Họ và tên thành viên:** [Điền tên Thành viên 2]
- **Vai trò chuyên trách:** Kỹ sư Thuật toán DRL (Deep Reinforcement Learning Algorithm Engineer).
- **Trách nhiệm cốt lõi:**
  1. Nghiên cứu, lựa chọn và thiết kế kiến trúc mạng nơ-ron tích chập (Nature CNN kết hợp Dueling Head) bằng **PyTorch**.
  2. Lập trình thuật toán cốt lõi **Dueling Double Deep Q-Network (Dueling DDQN)** hoàn chỉnh từ đầu.
  3. Xây dựng cấu trúc dữ liệu Bộ nhớ kinh nghiệm chuyển tiếp (**Replay Buffer**) dung lượng $100,000$ transitions tối ưu hóa bộ nhớ RAM.
  4. Triển khai cơ chế cân bằng Khám phá - Khai thác ($\epsilon$-Greedy Decay) và cơ chế cập nhật Mạng đích (Target Network Update).
  5. Thiết kế quy trình tính toán hàm mất mát (Huber Loss / Smooth L1 Loss) và lan truyền ngược tối ưu bằng Adam.
  6. Xây dựng module thuật toán đối sánh **PPO (Proximal Policy Optimization)** phục vụ mục tiêu lấy 2.0 điểm cải tiến mở rộng.
  7. Soạn thảo các chương liên quan đến Thuật toán và Lý thuyết trong Báo cáo học thuật (Chương 2, 4, 8) và thuyết trình phần tương ứng trên Slide.

---

## II. CƠ SỞ TOÁN HỌC & KIẾN TRÚC THUẬT TOÁN

### 1. Phương trình Bellman & Hiện tượng Overestimation Bias
Trong học Q-Learning truyền thống, hàm giá trị hành động tối ưu $Q^*(s, a)$ thỏa mãn phương trình Bellman tối ưu:
$$Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a \right]$$

Trong mạng DQN tiêu chuẩn (Mnih et al., 2015), mục tiêu cập nhật là:
$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a; \theta_t^-)$$

**Vấn đề:** Do cùng sử dụng bộ trọng số $\theta_t^-$ để vừa tìm hành động tốt nhất vừa đánh giá giá trị của nó, nếu có nhiễu ngẫu nhiên làm thổi phồng giá trị của một hành động bất kỳ, toán tử $\max$ sẽ luôn chọn giá trị bị thổi phồng đó. Qua hàng ngàn lượt cập nhật, giá trị Q bị ước lượng quá cao (Overestimation Bias), dẫn đến mạng tin tưởng sai lầm vào các hành động không thực sự tốt.

### 2. Thuật toán Double DQN (Van Hasselt et al., 2016)
Double DQN tách rời hai công đoạn:
1. **Lựa chọn hành động (Action Selection):** Sử dụng Mạng trực tuyến (Online Network $\theta_t$):
   $$a^* = \arg\max_a Q(S_{t+1}, a; \theta_t)$$
2. **Đánh giá hành động (Action Evaluation):** Sử dụng Mạng đích (Target Network $\theta_t^-$):
   $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q(S_{t+1}, a^*; \theta_t^-) = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$

### 3. Kiến trúc Dueling Network (Wang et al., 2016)
Giá trị $Q(s, a)$ được phân rã thành:
- $V(s)$: Giá trị nội tại của việc ở trạng thái $s$.
- $A(s, a)$: Mức độ ưu thế của hành động $a$ so với hành động trung bình tại $s$, với $\mathbb{E}[A(s, a)] = 0$.

Để đảm bảo tính xác định duy nhất (Identifiability), công thức chuẩn hóa trung bình được áp dụng:
$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|A|} \sum_{a' \in A} A(s, a'; \theta, \alpha) \right)$$
Trong đó:
- $\theta$: Trọng số các tầng tích chập trích xuất đặc trưng (Convolutional Backbone).
- $\beta$: Trọng số tầng Fully Connected ước lượng luồng Giá trị $V(s)$ (đầu ra scalar kích thước 1).
- $\alpha$: Trọng số tầng Fully Connected ước lượng luồng Lợi thế $A(s, a)$ (đầu ra vector kích thước $|A| = 7$).

---

## III. MÃ NGUỒN CHUẨN MỰC TRIỂN KHAI BẰNG PYTORCH

### 1. Kiến trúc mạng: `src/models/dqn_network.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Kiến trúc Nature CNN kết hợp Dueling Head cho Super Mario Bros.
    Đầu vào: Tensor (Batch, 4, 84, 84) đã chuẩn hóa về [0.0, 1.0].
    Đầu ra: Tensor Q-values (Batch, num_actions).
    """
    def __init__(self, in_channels=4, num_actions=7):
        super(DuelingDQN, self).__init__()
        
        # 1. Feature Extractor (Backbone Convolutional Layers)
        self.feature_network = nn.Sequential(
            # Conv 1: 4 -> 32 filters, kernel 8x8, stride 4, padding 0 -> Out: (32, 20, 20)
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv 2: 32 -> 64 filters, kernel 4x4, stride 2, padding 0 -> Out: (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv 3: 64 -> 64 filters, kernel 3x3, stride 1, padding 0 -> Out: (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() # 64 * 7 * 7 = 3136
        )
        
        # 2. Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 3. Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state):
        # Chuẩn hóa giá trị pixel từ [0, 255] về [0.0, 1.0] trực tiếp trên GPU
        state_normalized = state.float() / 255.0
        features = self.feature_network(state_normalized)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values
```

### 2. Bộ nhớ kinh nghiệm chuyển tiếp: `src/utils/replay_buffer.py`
```python
import numpy as np
import torch

class ReplayBuffer:
    """
    Bộ nhớ vòng tròn (Circular Buffer) lưu trữ kinh nghiệm dạng numpy uint8 để tiết kiệm RAM.
    Dung lượng: 100,000 transitions chỉ tốn khoảng ~2.8 GB RAM.
    """
    def __init__(self, capacity=100000, state_shape=(4, 84, 84), device='cuda'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Khởi tạo trước bộ nhớ cố định tránh phân mảnh RAM
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=64):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
```

### 3. Tác nhân Dueling Double DQN: `src/agents/double_dqn.py`
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.dqn_network import DuelingDQN
from src.utils.replay_buffer import ReplayBuffer

class DoubleDQNAgent:
    def __init__(self, in_channels=4, num_actions=7, lr=1e-4, gamma=0.99, 
                 buffer_capacity=100000, batch_size=64, target_update_freq=10000, device='cuda'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo Online Network và Target Network
        self.online_net = DuelingDQN(in_channels, num_actions).to(self.device)
        self.target_net = DuelingDQN(in_channels, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Target net chỉ dùng tính toán forward, không tính gradient
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # Huber Loss ổn định hơn MSE
        self.memory = ReplayBuffer(capacity=buffer_capacity, device=self.device)
        self.learn_step_counter = 0

    def select_action(self, state, epsilon):
        """
        Chọn hành động theo chiến lược Epsilon-Greedy.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            return q_values.argmax(dim=-1).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 1. Dự đoán Q-value hiện tại: Q(s, a; theta)
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Tính toán Double Q-Target:
        with torch.no_grad():
            # Bước A: Dùng Online Net để chọn hành động tốt nhất ở s_{t+1}: a* = argmax Q(s', a; theta)
            next_state_actions = self.online_net(next_states).argmax(dim=-1, keepdim=True)
            # Bước B: Dùng Target Net để đánh giá giá trị của a*: Q(s', a*; theta^-)
            next_q = self.target_net(next_states).gather(1, next_state_actions).squeeze(1)
            # Bước C: Target Bellman: y = r + gamma * next_q * (1 - done)
            target_q = rewards + (1.0 - dones.float()) * self.gamma * next_q
            
        # 3. Tính Huber Loss và Lan truyền ngược
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping để tránh bùng nổ gradient
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 4. Cập nhật Mạng đích định kỳ (Hard Update)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return loss.item()

    def save_checkpoint(self, filepath):
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
```

---

## IV. LỘ TRÌNH THỰC HIỆN CHI TIẾT THEO TỪNG TUẦN (WEEKLY SPRINT)

| Tuần | Mục tiêu tuần | Danh sách đầu việc cụ thể (Action Items) | File tạo ra / Chỉnh sửa | Bàn giao cho ai | Tiêu chí hoàn thành (DoD) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Tuần 1** | Thiết kế kiến trúc Mạng PyTorch | - Nghiên cứu tài liệu DeepMind Atari và Dueling DQN.<br>- Lập trình `src/models/dqn_network.py`.<br>- Viết unit test kiểm tra tensor forward qua 3 tầng Conv và Dueling Head.<br>- Đảm bảo tương thích input shape từ Người 1. | `src/models/dqn_network.py`<br>`tests/test_model.py` | Người 1, 4 | - Pytest vượt qua 100% test shapes.<br>- Mô hình chuyển đổi linh hoạt CPU/CUDA. |
| **Tuần 2** | Triển khai Thuật toán & Pipeline Huấn luyện | - Lập trình `ReplayBuffer` và `DoubleDQNAgent`.<br>- Cài đặt Huber Loss, gradient clipping và target network hard update.<br>- Phối hợp Người 3 viết script `train.py`.<br>- Soạn thảo Chương 2 (Cơ sở lý thuyết RL) và Chương 4 (Thiết kế hệ thống). | `src/agents/double_dqn.py`<br>`src/utils/replay_buffer.py`<br>`train.py`<br>`docs/chapter2.md`<br>`docs/chapter4.md` | Người 3, 4 | - Vòng lặp train 10k steps chạy mượt mà không lỗi shape.<br>- Loss tính toán chính xác, không bị `NaN`. |
| **Tuần 3** | Tối ưu hóa Thuật toán & Hỗ trợ Train 8 điểm | - Tinh chỉnh tham số học: Learning rate ($1\times 10^{-4}$), Batch size (64), Target update frequency ($10,000$ steps).<br>- Giám sát hiện tượng bùng nổ Q-values hoặc sụp đổ gradient.<br>- Lưu trữ checkpoint đạt mốc World 1-1 (`best_model.pth`). | `train.py`<br>`checkpoints/best_model.pth` | Người 3, 4 | - Thuật toán ổn định sau $2.0\times 10^6$ steps.<br>- Đạt mục tiêu chính 8 điểm (chạm cờ 1-1). |
| **Tuần 4** | Triển khai Thuật toán đối sánh PPO (2 điểm cộng) | - Xây dựng module thuật toán Actor-Critic PPO (`src/agents/ppo_agent.py`) hoặc thiết lập pipeline đối sánh bằng Stable-Baselines3.<br>- Huấn luyện PPO cùng số steps với DDQN để so sánh tốc độ hội tụ.<br>- Soạn thảo Chương 8 (Nghiên cứu mở rộng & Đối sánh thuật toán). | `src/agents/ppo_agent.py`<br>`configs/ppo_config.yaml`<br>`docs/chapter8.md` | Người 3, 4 | - Có bảng số liệu đối sánh rõ ràng giữa PPO và Dueling DDQN.<br>- Hoàn thành bản thảo Chương 8. |
| **Tuần 5** | Refactor Code, Viết Báo cáo & Luyện tập Slide | - Refactor toàn bộ mã nguồn thuật toán, bổ sung Type Hints và docstrings chuẩn PEP8.<br>- Kiểm tra lại các công thức toán học trong Báo cáo (Chương 2, 4, 8).<br>- Thiết kế các slide về Thuật toán Dueling DDQN và Target Network.<br>- Tham gia 2 buổi Mock Defense. | Mã nguồn sạch,<br>Slide phần 2,<br>Nội dung thuyết trình | Cả nhóm | - Trả lời lưu loát các câu hỏi toán học về Bellman, DDQN, Dueling.<br>- Thuyết trình đúng 5 phút. |

---

## V. NHỮNG LỖI THƯỜNG GẶP VÀ CÁCH KHẮC PHỤC TRIỆT ĐỂ (TROUBLESHOOTING)

1. **Lỗi `RuntimeError: CUDA out of memory` trong quá trình huấn luyện:**
   - *Nguyên nhân:* Lưu trực tiếp các tensor PyTorch vào Replay Buffer khiến GPU bị đầy bộ nhớ sau vài ngàn bước, hoặc batch size quá lớn ($> 128$).
   - *Giải pháp:* Replay Buffer **chỉ lưu trữ mảng Numpy `uint8` trên RAM của hệ thống** (dung lượng 100k transitions chỉ tốn ~2.8 GB RAM). Chỉ khi gọi hàm `sample()`, batch 64 mẫu mới được chuyển đổi sang Tensor và đẩy lên GPU (`.to(device)`).
2. **Hiện tượng giá trị Loss biến thành `NaN` hoặc Q-values tăng vô hạn:**
   - *Nguyên nhân:* Tốc độ học (learning rate) quá cao ($> 1\times 10^{-3}$), sử dụng MSE Loss bị khuếch đại bởi các lỗi dự đoán lớn, hoặc gradient bị bùng nổ (exploding gradient).
   - *Giải pháp:*
     - Chuyển từ `MSELoss` sang `SmoothL1Loss` (Huber Loss).
     - Bắt buộc áp dụng `torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)`.
     - Giữ learning rate của Adam trong khoảng $[1\times 10^{-4}, 2.5\times 10^{-4}]$.
3. **Mạng nơ-ron không học được gì (Loss giảm nhưng Mario vẫn đứng yên/nhảy tại chỗ):**
   - *Nguyên nhân:* Chưa detach tensor khi tính toán target Bellman (`target_q`), dẫn đến gradient bị truyền ngược sang cả Mạng đích (Target Network), phá vỡ tính ổn định của bài toán xấp xỉ hồi quy.
   - *Giải pháp:* Luôn bọc toàn bộ khối tính toán mục tiêu trong khối `with torch.no_grad():` như đã trình bày trong code template mục III.3.

---

## VI. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 2

- **Câu hỏi 1: Tại sao trong công thức Dueling DQN phải trừ đi giá trị trung bình $\frac{1}{|A|} \sum_{a'} A(s, a')$ thay vì chỉ lấy $Q(s, a) = V(s) + A(s, a)$?**
  - *Trả lời:* Nếu chỉ viết $Q(s, a) = V(s) + A(s, a)$, hệ thống sẽ rơi vào tình trạng **bất định (Unidentifiable)**. Nghĩa là với một giá trị $Q(s, a)$ cho trước, có vô số cặp nghiệm $(V, A)$ thỏa mãn (ví dụ: cộng thêm một hằng số $C$ vào $V$ và trừ $C$ khỏi $A$ thì $Q$ vẫn không đổi). Điều này khiến mạng nơ-ron không biết phải tối ưu luồng nào. Bằng cách trừ đi trung bình của các $A(s, a')$, ta ép giá trị ưu thế trung bình của các hành động bằng 0: $\frac{1}{|A|} \sum_{a'} \left( A(s, a') - \bar{A} \right) = 0$. Khi đó, tại trạng thái $s$, giá trị $V(s)$ phản ánh chính xác kỳ vọng giá trị trung bình của trạng thái đó, giúp việc tối ưu hóa hai luồng diễn ra độc lập và hội tụ ổn định.
- **Câu hỏi 2: Tại sao phải sử dụng Huber Loss (Smooth L1 Loss) thay vì Mean Squared Error (MSE)?**
  - *Trả lời:* Trong môi trường phức tạp như Mario, phần thưởng có thể biến động đột ngột (ví dụ khi chạm cờ được $+500$ hoặc bị chết $-50$). Khi sai số dự đoán $\delta = Q(s, a) - Y$ lớn, hàm mất mát MSE bình phương sai số ($\delta^2$) sẽ tạo ra gradient khổng lồ tỷ lệ thuận với $\delta$, dễ dẫn đến hiện tượng bùng nổ gradient và làm sụp đổ các trọng số mạng CNN đã học được trước đó. Huber Loss xử lý tuyệt vời bằng cách: ứng xử như hàm bậc hai $0.5 \delta^2$ khi $|\delta| \le 1.0$ (mượt mà ở gần điểm tối ưu), và ứng xử như hàm bậc một $|\delta| - 0.5$ khi $|\delta| > 1.0$ (gradient bị chặn ở mức hằng số $\pm 1$), bảo vệ mạng khỏi các cú sốc gradient cực đoan.
- **Câu hỏi 3: Sự khác biệt cơ bản giữa Double DQN (Off-policy) và PPO (On-policy) là gì?**
  - *Trả lời:* Double DQN là thuật toán Off-policy: tác nhân có thể học từ các dữ liệu kinh nghiệm trong quá khứ được lưu trữ trong Replay Buffer do các chính sách cũ tạo ra, nhờ đó hiệu quả sử dụng mẫu (Sample Efficiency) rất cao. Ngược lại, PPO là thuật toán On-policy: tác nhân chỉ học trực tiếp trên các quỹ đạo (trajectories) do chính sách hiện tại sinh ra và phải vứt bỏ dữ liệu đó ngay sau khi cập nhật gradient. Do đó, PPO đòi hỏi nhiều tương tác môi trường hơn nhưng lại có tính ổn định cao hơn và dễ tinh chỉnh siêu tham số hơn.

---

## VII. CHECKLIST TỰ ĐÁNH GIÁ TIẾN ĐỘ CỦA NGƯỜI 2

- [ ] Lập trình hoàn chỉnh class `DuelingDQN` với đúng 3 tầng Convolution và 2 luồng Value/Advantage.
- [ ] Chạy Pytest kiểm tra kích thước tensor đầu ra khớp hoàn toàn với 7 hành động.
- [ ] Replay Buffer vận hành trơn tru với $100,000$ transitions mà không gây rò rỉ RAM.
- [ ] Hàm mất mát Huber Loss và gradient clipping được cấu hình đúng chuẩn.
- [ ] Mạng đích (Target Network) được cập nhật định kỳ chính xác mỗi $10,000$ steps.
- [ ] Module thuật toán PPO đối sánh được hoàn thành và cho ra kết quả thực nghiệm.
- [ ] Hoàn thành bản thảo Chương 2, Chương 4 và Chương 8 của Báo cáo đồ án.
- [ ] Trình bày lưu loát phần thuyết trình thuật toán và trả lời 100% câu hỏi toán học DRL.
