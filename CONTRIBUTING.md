# Hướng dẫn đóng góp (CONTRIBUTING)

Tài liệu này quy định cách 4 thành viên làm việc chung trên repository. Quy ước
dựa trên mục 6.2 của [`KE_HOACH_TONG_THE_5_TUAN.md`](Phan_Cong_Nhiem_Vu/KE_HOACH_TONG_THE_5_TUAN.md).
Người 4 chịu trách nhiệm quản trị repository và có quyền phê duyệt kỹ thuật cuối cùng.

## 1. Mô hình nhánh (GitHub Flow)

Tạm thời nhóm **không dùng nhánh `develop`**: mọi nhánh tạo từ `main` và PR thẳng về `main`.

| Nhánh | Mục đích | Quy tắc |
| :--- | :--- | :--- |
| `main` | Nhánh chung, luôn chạy được | **Được bảo vệ.** Cấm push trực tiếp. Chỉ nhận thay đổi qua Pull Request. |
| `feature/<nguoi>-<mo-ta>` | Phát triển tính năng mới | Tạo từ `main`, PR về `main`. |
| `fix/<mo-ta>` | Sửa lỗi | Tạo từ `main`, PR về `main`. |
| `docs/<mo-ta>` | Tài liệu, báo cáo, slide | Tạo từ `main`, PR về `main`. |

Quy tắc đặt tên:

- Chữ thường, không dấu, nối bằng dấu gạch ngang `-`.
- `<nguoi>` là `n1`, `n2`, `n3`, `n4` (Người 1–4).
- Một nhánh chỉ giải quyết **một** tính năng/lỗi; xoá nhánh sau khi đã merge.

Ví dụ nhánh gợi ý cho từng người:

| Thành viên | Ví dụ nhánh |
| :--- | :--- |
| Người 1 – Môi trường | `feature/n1-env-wrappers`, `fix/framestack-shape` |
| Người 2 – Thuật toán | `feature/n2-dueling-dqn-agent`, `feature/n2-ppo` |
| Người 3 – Huấn luyện & Reward | `feature/n3-reward-shaping-v2`, `feature/n3-tensorboard-logging` |
| Người 4 – Tích hợp & Báo cáo | `feature/n4-eval-and-recording`, `docs/report-outline` |

## 2. Quy ước commit (Conventional Commits)

Cú pháp:

```text
<type>(<scope tuỳ chọn>): <mô tả ngắn, thể mệnh lệnh, <= 72 ký tự>

<thân bài tuỳ chọn: giải thích vì sao thay đổi>
```

| Type | Dùng khi | Ví dụ |
| :--- | :--- | :--- |
| `feat` | Thêm tính năng | `feat(env): add FrameStack wrapper for Gymnasium` |
| `fix` | Sửa lỗi | `fix(agent): resolve memory leak in replay buffer` |
| `docs` | Tài liệu, báo cáo | `docs: update chapter 3 of report` |
| `test` | Thêm/sửa test | `test: add unit test for tensor shape in dqn_network` |
| `refactor` | Tái cấu trúc, không đổi hành vi | `refactor(utils): extract seeding helper` |
| `chore` | Cấu hình, CI, phụ thuộc, dọn dẹp | `chore: pin torch version in requirements` |

Lưu ý:

- Mỗi commit là một thay đổi có ý nghĩa, chạy được; tránh commit kiểu `update`, `fix bug`.
- Không commit checkpoint (`*.pth`, `*.pt`), video (`*.mp4`), log TensorBoard (`runs/`)
  hay dữ liệu lớn. File lớn đưa lên Google Drive và dán link vào README/báo cáo.
- Không commit đường dẫn cứng tới máy cá nhân (`C:\Users\...`, `/home/...`).

## 3. Quy trình Pull Request

1. Cập nhật `main` mới nhất và tạo nhánh từ đó.
2. Code + viết test. Chạy toàn bộ test trên máy local:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest
   ```

3. Push nhánh và mở PR về `main`.
   Điền đầy đủ [PR template](.github/pull_request_template.md), ghi rõ số hạng mục WBS.
4. **Ít nhất 01 thành viên khác review và Approve.** Người 4 phê duyệt kỹ thuật cuối cùng
   khi có xung đột hoặc thay đổi ảnh hưởng nhiều module.
5. CI (`.github/workflows/tests.yml`) phải xanh. **Bắt buộc chạy `pytest` pass trước khi merge.**
6. Merge bằng **Squash and merge** (tiêu đề squash theo Conventional Commits), sau đó xoá nhánh.
7. Cuối tuần, Người 4 gắn tag trên `main` (vd. `v0.1.0`).

Khi có conflict: rebase/merge `main` mới nhất vào nhánh của mình, tự giải quyết,
chạy lại test rồi push. Nếu phức tạp, nhờ Người 4 hỗ trợ.

## 4. Bắt đầu: clone và tạo nhánh (cho cả 4 thành viên)

```powershell
# 1. Clone repository
git clone https://github.com/Eternity-KT/RL_Mario_Solving.git
cd RL_Mario_Solving

# 2. Cập nhật nhánh main
git switch main
git pull origin main

# 3. Cài môi trường (xem README.md, Python 3.13 64-bit)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest

# 4. Tạo nhánh làm việc của mình từ main
git switch -c feature/n2-dueling-dqn-agent

# 5. Làm việc, commit theo quy ước
git add src/agents/dqn_agent.py tests/test_agent.py
git commit -m "feat(agent): add Dueling Double DQN agent skeleton"

# 6. Đẩy nhánh lên và mở Pull Request về main
git push -u origin feature/n2-dueling-dqn-agent
```

Giữ nhánh luôn cập nhật với `main`:

```powershell
git fetch origin
git rebase origin/main        # hoặc: git merge origin/main
```

Trên Linux/macOS, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.

## 5. Cấu hình bảo vệ nhánh (Người 4 thực hiện trên GitHub)

Vào **Settings → Branches → Add branch protection rule** cho `main`:

- Require a pull request before merging (Require approvals: **1**).
- Require status checks to pass before merging → chọn job `pytest`.
- Do not allow bypassing the above settings; không cho phép force push / xoá nhánh.

## 6. Quy ước code ngắn gọn

- Python 3.13, PEP 8, docstring cho hàm/lớp public.
- Đường dẫn dùng `pathlib.Path` tương đối so với thư mục gốc repo hoặc nhận qua tham số/config.
- Siêu tham số đặt trong `configs/*.json`, không rải rác trong code.
- Đặt seed rõ ràng khi chạy thực nghiệm để đảm bảo tính tái lập.
- Code mới đặt đúng package: `src/env`, `src/models`, `src/agents`, `src/rewards`, `src/utils`.
