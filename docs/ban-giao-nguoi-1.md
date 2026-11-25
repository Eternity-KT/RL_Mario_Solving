# Bàn giao Người 1

Người 1 bàn giao phần môi trường và tiền xử lý cho Người 4 tích hợp vào pipeline PPO. Điểm vào là [`make_env()`](../mario_ppo/env.py#L31).

## Phần đã triển khai

| Wrapper | Đầu vào và đầu ra |
| --- | --- |
| `FrameSkip` | Nhận một hành động, lặp tối đa 4 khung theo cấu hình mặc định và dừng ngay khi lượt kết thúc hoặc bị cắt. Trả tổng reward véc-tơ `float32` của các khung đã chạy và ảnh lấy max theo từng pixel của hai khung cuối. Nếu chỉ chạy một khung thì dùng ảnh đó. |
| `GrayResize` | Đổi ảnh RGB `(240, 256, 3)` sang ảnh xám `(84, 84)` kiểu `uint8` theo độ sáng. `FrameStackObservation` trong `make_env()` ghép 4 ảnh thành quan sát `(4, 84, 84)` cho mạng. |
| `ScalarizeReward` | Tính tích vô hướng giữa trọng số và véc-tơ `[x_pos, time, death, coin, enemy]`. Trả reward kiểu `float` và giữ bản sao véc-tơ trong `info["vector_reward"]`. Trọng số mặc định là `[1.0, 1.0, 2.0, 0.05, 0.05]`. |
| `NoProgressTruncation` | Dùng `info["x_pos"]` để theo dõi tiến triển. Nếu Mario chưa vượt mốc gần nhất quá 2 pixel trong số bước cấu hình, wrapper đặt `truncated=True`. Khi môi trường tự kết thúc hoặc cắt lượt, wrapper giữ hai cờ do môi trường trả về. |

Ba wrapper đầu là phần bắt buộc của đề. `NoProgressTruncation` là tùy chọn của nhóm. [Mã wrapper](../mario_ppo/wrappers.py#L122)

## Cấu hình tích hợp

`max_no_progress_steps` mặc định bằng `0`, tức là tắt. Để thử cắt lượt sau 200 bước tác tử khi huấn luyện, đặt `env.max_no_progress_steps=200` khi chạy lệnh train, hoặc thêm vào cấu hình như sau. [Cấu hình mặc định](../configs/default.yaml#L21)

```yaml
env:
  max_no_progress_steps: 200
```

Wrapper cắt lượt nằm sau `FrameSkip` nên ngưỡng tính theo **bước tác tử**. `make_env()` chỉ thêm wrapper này trong nhánh huấn luyện. `RecordEpisodeStatistics` ghi tổng reward sau khi vô hướng hóa và trước khi `NormalizeReward` đổi thang cho PPO. [Chuỗi tích hợp](../mario_ppo/env.py#L40)

## Kiểm thử bàn giao

Chạy ở thư mục gốc bằng Python của môi trường đã cài các phụ thuộc dự án.

```bash
python -m pytest -q tests/test_wrappers.py
python -m pytest -q tests/test_pipeline.py -k "parse_level or make_env or eval_env"
```

- **11/11** bài kiểm thử wrapper đạt, gồm 7 bài có sẵn và 4 bài bổ sung. [Bài kiểm thử wrapper](../tests/test_wrappers.py)
- **3/3** bài kiểm thử pipeline liên quan đến tạo môi trường và chế độ đánh giá đạt. Kiểm tra trực tiếp trên trình giả lập ở màn `1-1` và `1-2` xác nhận quan sát `(4, 84, 84)` kiểu `uint8` và cắt lượt do kẹt chỉ bật trong chế độ huấn luyện. [Bài kiểm thử pipeline](../tests/test_pipeline.py)
- Bộ kiểm thử đầy đủ có **32 bài đạt, 7 bài lỗi** tại thời điểm bàn giao. Các lỗi thuộc nhóm kiểm thử model, huấn luyện và đánh giá vì `ActorCritic` trên `main` còn TODO. [Mã ActorCritic](../mario_ppo/model.py#L30)
