# SỔ TAY KỸ THUẬT & QUY TRÌNH LÀM VIỆC CHI TIẾT - NGƯỜI 4
## VAI TRÒ: CHUYÊN GIA TÍCH HỢP HỆ THỐNG, KIỂM THỬ, ĐÁNH GIÁ ĐỊNH LƯỢNG, TRỰC QUAN HÓA & TRƯỞNG NHÓM HỌC THUẬT (SYSTEM INTEGRATION, QA, BENCHMARK & ACADEMIC COORDINATOR)

---

## I. THÔNG TIN CHUNG & VAI TRÒ TRÁCH NHIỆM

- **Họ và tên thành viên:** [Điền tên Thành viên 4]
- **Vai trò chuyên trách:** Kỹ sư Tích hợp Hệ thống & Đảm bảo Chất lượng (System Integration & QA Engineer) kiêm Trưởng nhóm Điều phối Học thuật.
- **Trách nhiệm cốt lõi:**
  1. **Đóng góp kỹ thuật mã nguồn độc lập:**
     - Thiết lập cấu trúc Repository chuẩn quốc tế, quy ước phân nhánh Git Flow và quản trị hợp nhất mã nguồn (Pull Requests).
     - Lập trình bộ kiểm thử tự động với **Pytest** (`tests/`) kiểm tra tính tương thích giữa môi trường (Người 1) và mô hình (Người 2).
     - Xây dựng module đánh giá định lượng độc lập (`evaluate.py`) đo lường Win Rate, Max X-Position trên 20 seeds ngẫu nhiên.
     - Lập trình ứng dụng chạy Demo trực quan thời gian thực (`demo.py`) kèm thanh thông số HUD (Q-values, Actions, X-pos) và tự động xuất video MP4 full-HD.
     - Thực hiện kiểm thử tính tái lập (Reproducibility Testing) từ môi trường máy tính trắng.
  2. **Trách nhiệm học thuật và điều phối chung:**
     - Lập dàn ý chi tiết, phân bổ nhiệm vụ viết báo cáo cho từng người và trực tiếp chịu trách nhiệm tổng hợp, biên tập chuẩn hóa bản cuối (Chương 10, Mở đầu, Kết luận, Tài liệu tham khảo chuẩn IEEE).
     - Thiết kế bộ Slide thuyết trình chuẩn mực, phân chia thời lượng nói và điều phối các buổi diễn tập bảo vệ (Mock Defense).

---

## II. ĐÓNG GÓP KỸ THUẬT: THIẾT KẾ MÃ NGUỒN VÀ BỘ CÔNG CỤ ĐÁNH GIÁ

### 1. Script đánh giá định lượng khoa học: `evaluate.py`
Người 4 chịu trách nhiệm viết file `evaluate.py` để chứng minh một cách khách quan năng lực giải quyết màn World 1-1 của tác nhân:

```python
import argparse
import torch
import numpy as np
import pandas as pd
from src.env.make_env import make_mario_env
from src.agents.double_dqn import DoubleDQNAgent

def evaluate_agent(checkpoint_path, num_episodes=20, render=False, output_csv="results/eval_results.csv"):
    """
    Đánh giá tác nhân một cách hoàn toàn khách quan:
    - Epsilon = 0.0 (Hoàn toàn khai thác chính sách tối ưu, không có hành động ngẫu nhiên).
    - Chạy trên danh sách các seeds ngẫu nhiên được cố định trước để đảm bảo tính tái lập.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = make_mario_env(world=1, stage=1, action_type='SIMPLE')
    
    agent = DoubleDQNAgent(in_channels=4, num_actions=7, device=device)
    agent.load_checkpoint(checkpoint_path)
    agent.online_net.eval()
    
    seeds = [42, 100, 2024, 777, 999, 1234, 5555, 8888, 1111, 2222,
             3333, 4444, 6666, 7777, 1010, 2020, 3030, 4040, 5050, 9999]
    
    results = []
    wins = 0
    
    print(f"[*] Bắt đầu đánh giá mô hình: {checkpoint_path} trên {num_episodes} episodes độc lập...")
    
    for i in range(min(num_episodes, len(seeds))):
        seed = seeds[i]
        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        steps = 0
        max_x = info.get('x_pos', 40)
        
        while not done:
            action = agent.select_action(obs, epsilon=0.0) # Thuần khai thác
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            x_pos = info.get('x_pos', max_x)
            if x_pos > max_x:
                max_x = x_pos
                
            if terminated or truncated:
                done = True
                
        flag_get = info.get('flag_get', False)
        if flag_get:
            wins += 1
            
        print(f"Episode {i+1:02d} (Seed {seed}): Max X = {max_x}, Steps = {steps}, Win = {flag_get}")
        results.append({
            'episode': i + 1,
            'seed': seed,
            'max_x': max_x,
            'steps': steps,
            'total_reward': total_reward,
            'win': flag_get
        })
        
    env.close()
    
    # Tính toán các chỉ số thống kê tổng hợp
    df = pd.DataFrame(results)
    win_rate = (wins / num_episodes) * 100.0
    avg_max_x = df['max_x'].mean()
    avg_steps = df['steps'].mean()
    
    print("="*60)
    print(f"KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP ({num_episodes} Episodes):")
    print(f"- Tỷ lệ hoàn thành màn chơi (Win Rate): {win_rate:.2f}%")
    print(f"- Tọa độ X cực đại trung bình: {avg_max_x:.2f} (Cột cờ: X >= 3161)")
    print(f"- Số bước trung bình: {avg_steps:.2f}")
    print("="*60)
    
    df.to_csv(output_csv, index=False)
    print(f"[+] Kết quả chi tiết đã được lưu tại: {output_csv}")
    return win_rate, avg_max_x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate_agent(args.checkpoint, args.episodes)
```

### 2. Script Demo trực quan & Tự động xuất Video MP4 kèm HUD: `demo.py`
Người 4 xây dựng file `demo.py` để trình chiếu trực tiếp trong buổi bảo vệ đồ án:

```python
import cv2
import torch
import numpy as np
from src.env.make_env import make_mario_env
from src.agents.double_dqn import DoubleDQNAgent

ACTION_NAMES = ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'A', 'Left']

def draw_hud(frame, action, q_values, x_pos, step):
    """
    Vẽ bảng điều khiển Heads-up Display (HUD) chuyên nghiệp lên khung hình video:
    Hiển thị phím bấm của AI, tọa độ X thời gian thực và thanh biểu đồ Q-values.
    """
    h, w, _ = frame.shape
    # Tạo thanh banner bán trong suốt ở mép trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Hiển thị chữ
    text_info = f"Step: {step} | Pos X: {x_pos} | Action: {ACTION_NAMES[action]}"
    cv2.putText(frame, text_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Hiển thị giá trị Q tối đa
    best_q = np.max(q_values)
    cv2.putText(frame, f"Max Q-value: {best_q:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def run_demo(checkpoint_path, record_mp4=True, output_video="recordings/mario_victory_demo.mp4"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = make_mario_env(world=1, stage=1, action_type='SIMPLE')
    
    agent = DoubleDQNAgent(in_channels=4, num_actions=7, device=device)
    agent.load_checkpoint(checkpoint_path)
    agent.online_net.eval()
    
    obs, info = env.reset(seed=42)
    done = False
    step = 0
    frames = []
    
    print("[*] Đang khởi chạy Demo trực quan Super Mario Bros...")
    
    while not done:
        # Lấy khung hình RGB gốc từ NES để render sắc nét
        raw_rgb = env.unwrapped.render()
        
        # Dự đoán hành động và Q-values
        state_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = agent.online_net(state_tensor).cpu().numpy()[0]
        action = np.argmax(q_vals)
        
        # Bước đi trong môi trường
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        x_pos = info.get('x_pos', 0)
        
        # Vẽ HUD lên khung hình
        frame_hud = draw_hud(raw_rgb.copy(), action, q_vals, x_pos, step)
        frames.append(frame_hud)
        
        if terminated or truncated:
            done = True
            
    env.close()
    print(f"[+] Episode kết thúc tại Step {step}, X-pos = {x_pos}, Flag = {info.get('flag_get', False)}")
    
    if record_mp4 and len(frames) > 0:
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))
        for f in frames:
            # OpenCV ghi màu theo chuẩn BGR
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[+] Video kết quả đã được lưu tại: {output_video}")

if __name__ == "__main__":
    run_demo("checkpoints/best_model.pth")
```

---

## III. LỘ TRÌNH THỰC HIỆN CHI TIẾT THEO TỪNG TUẦN (WEEKLY SPRINT)

| Tuần | Mục tiêu tuần | Danh sách đầu việc cụ thể (Action Items) | File tạo ra / Chỉnh sửa | Bàn giao cho ai | Tiêu chí hoàn thành (DoD) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Tuần 1** | Khởi tạo Quản trị Dự án & Dàn ý Báo cáo | - Tạo GitHub repo, thiết lập `.gitignore`, quy ước branch và PR template.<br>- Viết file `tests/test_env.py` kiểm thử môi trường của Người 1.<br>- Lập dàn ý Báo cáo đồ án 12 chương và phân công từng mục cho 4 người. | `.gitignore`<br>`README.md`<br>`tests/test_env.py`<br>`docs/outline.md` | Cả nhóm | - Khung repo hoàn chỉnh.<br>- 4 thành viên clone và tạo branch thành công.<br>- Dàn ý báo cáo được thông qua. |
| **Tuần 2** | Kiểm thử Tích hợp Alpha & Điều phối Báo cáo | - Viết unit test kiểm tra tính tương thích giữa môi trường và mạng nơ-ron (`tests/test_pipeline.py`).<br>- Hỗ trợ Người 2 và 3 đóng gói script `train.py`.<br>- Thu thập bản thảo Chương 1, 2, 3 từ Người 1, 2; rà soát format. | `tests/test_pipeline.py`<br>`docs/draft_ch1_3.docx` | Người 1, 2 | - Kiểm thử tích hợp chạy thành công.<br>- Hoàn thiện bản thảo sơ bộ Chương 1-3. |
| **Tuần 3** | Xây dựng Bộ Đánh giá Độc lập & Nghiệm thu 8 điểm | - Lập trình script đánh giá định lượng khoa học `evaluate.py`.<br>- Thực hiện đánh giá chéo checkpoint `best_model.pth` của Người 3 trên 20 seeds ngẫu nhiên.<br>- Thu thập và biên tập Chương 4, 5, 6 của Báo cáo. | `evaluate.py`<br>`results/eval_results.csv`<br>`docs/draft_ch4_6.docx` | Cả nhóm | - Xác nhận khách quan: Win Rate $\ge 80\%$, $X \ge 3161$.<br>- Nghiệm thu chính thức mốc 8.0 điểm. |
| **Tuần 4** | Xây dựng Ứng dụng Demo HUD & Kiểm thử Tái lập | - Lập trình `demo.py` hiển thị HUD trực quan và xuất file video MP4.<br>- Kiểm thử tính tái lập (Reproducibility): Cài đặt từ đầu trên máy tính thứ hai, kiểm tra không bị lỗi hard-coded path.<br>- Viết Chương 10 (Hướng dẫn cài đặt & Kiểm thử). | `demo.py`<br>`recordings/*.mp4`<br>`docs/chapter10.md` | Người 1, 3 | - Xuất được video demo MP4 sắc nét.<br>- Xác nhận chạy thành công trên máy tính thứ hai. |
| **Tuần 5** | Xuất bản Báo cáo, Hoàn thiện Slide & Điều phối Diễn tập | - Khóa toàn bộ mã nguồn (Code Freeze), tạo Release v1.0.0.<br>- Hoàn thiện file Báo cáo toàn diện (Chương 1-12, trích dẫn IEEE, format Times New Roman).<br>- Thiết kế bộ Slide 25-28 trang chuyên nghiệp.<br>- Điều phối và bấm giờ 02 buổi diễn tập bảo vệ thử nghiệm. | `Bao_cao_RL_Mario.pdf`<br>`Slide_Thuyet_trinh.pptx`<br>`Release v1.0.0` | Giảng viên & Hội đồng | - Sản phẩm đồ án đạt chuẩn xuất sắc 10/10.<br>- Cả nhóm sẵn sàng 100% tự tin bảo vệ. |

---

## IV. BẢN HƯỚNG DẪN BIÊN TẬP VÀ ĐỊNH DẠNG BÁO CÁO HỌC THUẬT

Người 4 chịu trách nhiệm kỹ thuật định dạng văn bản (Format Lead), bắt buộc kiểm tra các tiêu chuẩn sau trước khi xuất file:
1. **Khổ giấy & Căn lề chuẩn học thuật:**
   - Khổ giấy: A4 ($210 \times 297$ mm).
   - Căn lề: Top: $2.5$ cm, Bottom: $2.5$ cm, Left: $3.0$ cm (để đóng gáy), Right: $2.0$ cm.
2. **Quy chuẩn Font chữ và Giãn dòng:**
   - Font chữ: **Times New Roman**.
   - Cỡ chữ nội dung (Body text): 13 pt, giãn dòng $1.3 - 1.5$ lines, thụt đầu dòng đoạn văn $1.0$ cm.
   - Heading 1: 18 pt, Đậm, viết hoa.
   - Heading 2: 15 pt, Đậm.
   - Heading 3: 13 pt, Đậm hoặc Nghiêng.
3. **Quy chuẩn Bảng biểu và Hình vẽ:**
   - Tất cả hình vẽ đều phải có chú thích bên dưới (ví dụ: *Hình 4.2: Kiến trúc mạng Dueling Convolutional Neural Network*).
   - Tất cả bảng biểu đều phải có chú thích bên trên (ví dụ: *Bảng 7.1: Bảng kết quả đánh giá thực nghiệm trên 20 seeds độc lập*).
   - Trích dẫn tài liệu tham khảo theo đúng chuẩn quốc tế **IEEE** (`[1]`, `[2]`).

---

## V. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 4

- **Câu hỏi 1: Tại sao trong quá trình đánh giá bằng script `evaluate.py`, nhóm phải tắt hoàn toàn tính ngẫu nhiên ($\epsilon = 0.0$) và chạy trên 20 seeds khác nhau?**
  - *Trả lời:* Trong quá trình huấn luyện, $\epsilon > 0$ là bắt buộc để tác nhân thực hiện khám phá môi trường (Exploration). Tuy nhiên, khi đánh giá năng lực thực sự của mô hình đã học xong, ta bắt buộc phải đặt $\epsilon = 0.0$ (chế độ Pure Exploitation) để tác nhân luôn luôn lựa chọn hành động có giá trị kỳ vọng Q-value cao nhất theo đúng chính sách tối ưu. Việc chạy trên 20 seeds ngẫu nhiên khác nhau là phương pháp chuẩn mực trong nghiên cứu học thuật để loại bỏ yếu tố may rủi, kiểm tra khả năng ứng phó ổn định của tác nhân trước các biến động vi mô của môi trường game (như thời điểm xuất hiện và tương tác ngẫu nhiên của Goomba).
- **Câu hỏi 2: Nhóm đã làm thế nào để đảm bảo tính tái lập (Reproducibility) của sản phẩm mã nguồn?**
  - *Trả lời:* Tính tái lập được nhóm bảo đảm thông qua 4 cơ chế chặt chẽ:
    1. *Cố định hạt giống ngẫu nhiên (Fixed Seeds):* Đặt seed đồng nhất cho Python `random`, `numpy`, và `torch` (cả CPU lẫn `torch.cuda.manual_seed_all`).
    2. *Đóng băng phiên bản thư viện:* File `requirements.txt` ghim chính xác phiên bản của từng package, tránh hiện tượng thay đổi API ngầm định.
    3. *Không sử dụng đường dẫn tuyệt đối (No Hard-coded paths):* Toàn bộ mã nguồn sử dụng đường dẫn tương đối thông qua module `pathlib` hoặc `os.path`.
    4. *Thực nghiệm trên môi trường sạch:* Em đã trực tiếp clone mã nguồn từ GitHub sang một máy tính mới hoàn toàn và chạy thành công quy trình đánh giá và demo theo đúng tài liệu `README.md`.
- **Câu hỏi 3: Phân tích sự đánh đổi giữa thời gian tính toán và độ mượt mà khi hiển thị giao diện chơi của Mario trong script `demo.py`?**
  - *Trả lời:* Môi trường huấn luyện sử dụng kỹ thuật bỏ qua khung hình (FrameSkip = 4) trên ảnh xám $84 \times 84$ để tối ưu hóa tốc độ. Tuy nhiên, nếu render trực tiếp ảnh này cho con người xem thì chuyển động sẽ bị giật cục và mờ. Trong script `demo.py`, em đã bọc cơ chế render thông minh: vẫn đưa tensor $84 \times 84$ vào mạng nơ-ron để ra quyết định với tần số 15 Hz, nhưng trích xuất khung hình màu RGB gốc $256 \times 240$ trực tiếp từ bộ nhớ đệm của giả lập NES ở tần số 30-60 FPS, vẽ thanh thông số HUD trực quan bằng OpenCV rồi ghi vào video MP4. Nhờ đó, người xem được thưởng thức một video gameplay mượt mà, độ nét cao mà thuật toán AI vẫn chạy hoàn toàn chính xác.

---

## VI. CHECKLIST TỰ ĐÁNH GIÁ TIẾN ĐỘ CỦA NGƯỜI 4

- [ ] Repository GitHub được cấu hình chuyên nghiệp với đầy đủ branches và README.
- [ ] Lập trình hoàn chỉnh script `evaluate.py` với ma trận đánh giá 20 seeds.
- [ ] Lập trình hoàn chỉnh script `demo.py` hiển thị HUD và tự động ghi video MP4.
- [ ] Đã kiểm thử tính tái lập thành công trên máy tính thứ hai độc lập.
- [ ] Báo cáo đồ án đạt chuẩn hình thức A4, font Times New Roman, đầy đủ 12 chương.
- [ ] Danh mục tài liệu tham khảo chuẩn IEEE với tối thiểu 10 bài báo chất lượng.
- [ ] Slide thuyết trình đạt chuẩn thẩm mỹ cao, khớp thời lượng 20 phút.
- [ ] Điều phối thành công 02 buổi diễn tập thuyết trình có bấm giờ.
- [ ] Trả lời xuất sắc các câu hỏi về tích hợp, kiểm thử và tính tái lập hệ thống.
