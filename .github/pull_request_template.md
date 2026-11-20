## Mô tả thay đổi

<!-- PR này làm gì? Vì sao cần? Tóm tắt 2-5 dòng. -->

## Liên kết công việc (WBS)

- Hạng mục WBS: #<!-- số hạng mục trong KE_HOACH_TONG_THE_5_TUAN.md, vd. 2 -->
- Người phụ trách: <!-- Người 1 / 2 / 3 / 4 -->
- Issue liên quan (nếu có): <!-- Closes #... -->

## Loại thay đổi

- [ ] `feat` – tính năng mới
- [ ] `fix` – sửa lỗi
- [ ] `docs` – tài liệu / báo cáo
- [ ] `test` – thêm/sửa kiểm thử
- [ ] `refactor` – tái cấu trúc, không đổi hành vi
- [ ] `chore` – cấu hình, CI, phụ thuộc

## Cách kiểm tra

<!-- Lệnh để reviewer tự chạy lại, vd: python -m pytest tests/test_wrappers.py tests/test_nhom_wrappers.py -->

## Checklist

- [ ] Branch đặt tên đúng quy ước (`feature/...`, `fix/...`, `docs/...`) và PR nhắm vào `main`
- [ ] Commit message theo Conventional Commits
- [ ] Đã chạy `python -m pytest` trên máy local và tất cả test đều pass
- [ ] Đã thêm/cập nhật test cho phần code mới (nếu có)
- [ ] KHÔNG commit checkpoint (`*.pth`, `*.pt`), video (`*.mp4`), log TensorBoard (`runs/`) hay file dữ liệu lớn
- [ ] KHÔNG có đường dẫn cứng (hard-coded path) kiểu `C:\Users\...` / `/home/...`; dùng đường dẫn tương đối hoặc tham số dòng lệnh / file config
- [ ] Không có seed, siêu tham số "ma thuật" bị sửa ngầm mà không ghi chú
- [ ] Đã cập nhật README / docs nếu thay đổi cách cài đặt hoặc cách chạy
- [ ] Đã yêu cầu ít nhất 01 thành viên khác review

## Ghi chú cho reviewer

<!-- Điểm cần chú ý, ảnh chụp / kết quả đo nếu có. -->
