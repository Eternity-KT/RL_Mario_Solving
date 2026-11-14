# GIẢI QUYẾT TRÒ CHƠI SUPER MARIO BROS (WORLD 1-1) BẰNG DEEP REINFORCEMENT LEARNING

> **Đồ án cuối kỳ môn học: Học tăng cường (Reinforcement Learning)**  
> **Quy mô nhóm:** 04 thành viên | **Thời gian thực hiện:** 05 tuần (35 ngày)  
> **Mục tiêu điểm số:** 10.0 / 10.0 (8.0đ Mục tiêu chính + 2.0đ Cải tiến mở rộng)

---

## 📌 HỆ THỐNG TÀI LIỆU KẾ HOẠCH & SỔ TAY KỸ THUẬT (MARKDOWN)

Toàn bộ kế hoạch triển khai, phân công chi tiết và mã nguồn mẫu được lưu trữ tập trung tại thư mục: [`Phan_Cong_Nhiem_Vu/`](file:///d:/University_Subject/RL/RL_Mario_Solving/Phan_Cong_Nhiem_Vu/)

1. 📘 [`KE_HOACH_TONG_THE_5_TUAN.md`](file:///d:/University_Subject/RL/RL_Mario_Solving/Phan_Cong_Nhiem_Vu/KE_HOACH_TONG_THE_5_TUAN.md) *(57 KB)*: Kế hoạch tổng thể dự án 5 tuần, mô hình toán học MDP, WBS 26 hạng mục công việc (P0/P1/P2), ma trận RACI, kế hoạch 12 chương báo cáo, kế hoạch slide, quản lý rủi ro và tiêu chí nghiệm thu.
2. 📗 [`NGUOI_1_MOI_TRUONG_VA_TIEN_XU_LY.md`](file:///d:/University_Subject/RL/RL_Mario_Solving/Phan_Cong_Nhiem_Vu/NGUOI_1_MOI_TRUONG_VA_TIEN_XU_LY.md) *(17 KB)*: Sổ tay kỹ thuật Người 1 (Môi trường Gymnasium, Custom Wrappers: Grayscale, Resize $84 \times 84$, FrameSkip = 4, FrameStack = 4, JoypadSpace, code template `make_mario_env()`, xử lý lỗi `nes-py`, ngân hàng câu hỏi vấn đáp riêng).
3. 📕 [`NGUOI_2_THUAT_TOAN_DQN_PPO.md`](file:///d:/University_Subject/RL/RL_Mario_Solving/Phan_Cong_Nhiem_Vu/NGUOI_2_THUAT_TOAN_DQN_PPO.md) *(21 KB)*: Sổ tay kỹ thuật Người 2 (Mạng PyTorch Nature CNN + Dueling Head, Double Q-learning, Replay Buffer lưu mảng `uint8` trên RAM, Huber Loss, code template `DoubleDQNAgent`, thuật toán đối sánh PPO).
4. 📙 [`NGUOI_3_HUAN_LUYEN_REWARD_THUC_NGHIEM.md`](file:///d:/University_Subject/RL/RL_Mario_Solving/Phan_Cong_Nhiem_Vu/NGUOI_3_HUAN_LUYEN_REWARD_THUC_NGHIEM.md) *(16 KB)*: Sổ tay kỹ thuật Người 3 (Lý thuyết Reward Shaping, khắc phục 4 bẫy hành vi: sợ quái vật nhảy lùi, kẹt ống nước cao; code template `MarioCustomRewardWrapper` v2 tích hợp Anti-stuck penalty; TensorBoard metrics; kịch bản Ablation Study 2 điểm).
5. 📓 [`NGUOI_4_TICH_HOP_TEST_BAO_CAO_DEMO.md`](file:///d:/University_Subject/RL/RL_Mario_Solving/Phan_Cong_Nhiem_Vu/NGUOI_4_TICH_HOP_TEST_BAO_CAO_DEMO.md) *(17 KB)*: Sổ tay kỹ thuật Người 4 (Git Flow, Pytest suite, code template `evaluate.py` 20 seeds ngẫu nhiên với $\epsilon = 0.0$, code template `demo.py` hiển thị HUD thời gian thực và ghi video MP4 full-HD, kiểm thử tính tái lập, điều phối Báo cáo IEEE và Slide).

> *Lưu ý: Nếu cần xuất file Word (.docx) để in ấn hoặc nộp cho giảng viên, chỉ cần chạy lệnh: `python generate_docx.py`.*

---

## 👥 PHÂN CHIA VAI TRÒ 4 THÀNH VIÊN

| STT | Thành viên | Vai trò chuyên trách | Nhiệm vụ kỹ thuật cốt lõi |
| :---: | :---: | :--- | :--- |
| **1** | **Người 1** | Kỹ sư Môi trường & Tiền xử lý | Môi trường Gymnasium, Custom Wrappers (84x84, Skip 4, Stack 4), Joypad 7 actions, baseline test |
| **2** | **Người 2** | Kỹ sư Thuật toán DRL & Mạng | Mạng PyTorch Dueling CNN, Double DQN, Replay Buffer uint8, đối sánh PPO |
| **3** | **Người 3** | Kỹ sư Huấn luyện & Thực nghiệm | Reward Shaping v2 (Anti-stuck), TensorBoard log, train 3M steps, Ablation Study |
| **4** | **Người 4** | Kỹ sư Tích hợp, QA & Báo cáo | Git Flow, Pytest, evaluate.py (20 seeds), demo.py HUD & MP4, Tổng tài Báo cáo |

---

## 🎯 MỤC TIÊU NGHIỆM THU CHÍNH (8.0 ĐIỂM + 2.0 ĐIỂM)
1. **Mục tiêu chính (8 điểm):** Mario hoàn thành màn **World 1-1** (`flag_get == True`, $X \ge 3161$), tỷ lệ thắng $\ge 80\%$ trên 20 test episodes độc lập ($\epsilon = 0.0$).
2. **Cải tiến mở rộng (2 điểm):** Nghiên cứu đối sánh Dueling DDQN vs PPO; Ablation Study về Reward Shaping; Ứng dụng Demo HUD thời gian thực kèm video MP4 full-HD.
