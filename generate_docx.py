# -*- coding: utf-8 -*-
"""
Script tự động tạo file Word: Phan_cong_cong_viec_nhom_Mario_Gymnasium_5_tuan.docx
Đạt chuẩn học thuật: Khổ A4, Font Times New Roman, Bảng biểu chuyên nghiệp, Đánh số trang, Màu sắc ưu tiên P0/P1/P2.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Đặt màu nền cho cell trong bảng"""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    """Đặt padding cho cell (đơn vị dxa, 20 dxa = 1 pt)"""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Đặt viền thanh lịch cho bảng"""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:insideV w:val="none"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def add_page_number_to_section(section):
    """Thêm số trang vào footer"""
    footer = section.footer
    f_p = footer.paragraphs[0]
    f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    f_run = f_p.add_run("Trang ")
    f_run.font.name = "Times New Roman"
    f_run.font.size = Pt(9.5)
    f_run.font.color.rgb = RGBColor(128, 128, 128)
    
    # Thêm trường PAGE động trong Word XML
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    f_p._p.append(fldSimple)
    
    f_run2 = f_p.add_run(" / ")
    f_run2.font.name = "Times New Roman"
    f_run2.font.size = Pt(9.5)
    f_run2.font.color.rgb = RGBColor(128, 128, 128)
    
    fldSimple2 = OxmlElement('w:fldSimple')
    fldSimple2.set(qn('w:instr'), 'NUMPAGES')
    f_p._p.append(fldSimple2)

def create_full_document(output_path):
    doc = Document()
    
    # 1. Cấu hình trang A4 và lề chuẩn học thuật
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0) # Lề gáy đóng tài liệu
        section.right_margin = Cm(2.0)
        
        # Header
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Đồ án Học tăng cường: Giải Super Mario Bros (Gymnasium/Python) | Nhóm 4 thành viên")
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(8.5)
        hrun.font.italic = True
        hrun.font.color.rgb = RGBColor(140, 140, 140)
        
        # Footer
        add_page_number_to_section(section)

    # Cấu hình Style mặc định
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(12)
    style_normal.font.color.rgb = RGBColor(30, 30, 30)
    style_normal.paragraph_format.line_spacing = 1.25
    style_normal.paragraph_format.space_after = Pt(4)

    # Màu sắc định danh
    NAVY = RGBColor(0, 51, 102)       # Heading 1
    BLUE = RGBColor(0, 102, 153)      # Heading 2
    DARK_GRAY = RGBColor(50, 50, 50)  # Heading 3
    
    # Hàm helper thêm heading chuẩn
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = NAVY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = BLUE
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = DARK_GRAY
        return p

    def add_p(text, bold_prefix="", italic_prefix=""):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(4)
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.font.bold = True
            r_b.font.name = 'Times New Roman'
        if italic_prefix:
            r_i = p.add_run(italic_prefix)
            r_i.font.italic = True
            r_i.font.name = 'Times New Roman'
        r_t = p.add_run(text)
        r_t.font.name = 'Times New Roman'
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(3)
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.font.bold = True
            r_b.font.name = 'Times New Roman'
        r_t = p.add_run(text)
        r_t.font.name = 'Times New Roman'
        return p

    def add_callout(text, title="LƯU Ý QUAN TRỌNG:"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        tbl.columns[0].width = Cm(16.0)
        c = tbl.cell(0, 0)
        set_cell_background(c, "F0F4F8")
        set_cell_margins(c, top=140, bottom=140, left=180, right=180)
        
        # Border trái màu xanh navy đậm
        tcPr = c._element.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:left w:val="single" w:sz="24" w:space="0" w:color="003366"/>'
            f'<w:top w:val="none"/>'
            f'<w:bottom w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)
        
        p = c.paragraphs[0]
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(2)
        r_t = p.add_run(f"[{title}] ")
        r_t.font.bold = True
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(11.5)
        r_t.font.color.rgb = NAVY
        
        r_b = p.add_run(text)
        r_b.font.name = 'Times New Roman'
        r_b.font.size = Pt(11)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # =========================================================================
    # TRANG BÌA (COVER PAGE)
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_after = Pt(2)
    r = p_inst.add_run("ĐẠI HỌC QUỐC GIA - KHOA CÔNG NGHỆ THÔNG TIN\nBỘ MÔN TRÍ TUỆ NHÂN TẠO & HỌC MÁY")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = DARK_GRAY

    p_line = doc.add_paragraph()
    p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_line.paragraph_format.space_after = Pt(40)
    r_l = p_line.add_run("________________________________________")
    r_l.font.color.rgb = BLUE

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(12)
    r_t = p_title.add_run("KẾ HOẠCH TRIỂN KHAI VÀ PHÂN CÔNG NHIỆM VỤ ĐỒ ÁN CUỐI KỲ\n")
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(18)
    r_t.font.bold = True
    r_t.font.color.rgb = NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(40)
    r_s = p_sub.add_run("ĐỀ TÀI: GIẢI TRÒ CHƠI SUPER MARIO BROS (WORLD 1-1)\nBẰNG DEEP REINFORCEMENT LEARNING VỚI GYMNASIUM VÀ PYTHON")
    r_s.font.name = 'Times New Roman'
    r_s.font.size = Pt(14)
    r_s.font.bold = True
    r_s.font.color.rgb = BLUE

    # Bảng thông tin nhóm trên trang bìa
    cover_tbl = doc.add_table(rows=7, cols=2)
    cover_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cover_tbl.autofit = False
    cover_tbl.columns[0].width = Cm(5.5)
    cover_tbl.columns[1].width = Cm(10.5)
    
    meta_data = [
        ("Môn học:", "Học tăng cường (Reinforcement Learning - RL)"),
        ("Quy mô nhóm:", "04 Thành viên chuyên trách"),
        ("Thời gian thực hiện:", "05 Tuần (35 ngày làm việc liên tục)"),
        ("Mục tiêu kỹ thuật:", "Giải hoàn toàn màn World 1-1 (Win Rate >= 80%)"),
        ("Mục tiêu điểm số:", "10.0 / 10.0 (8.0đ Mục tiêu chính + 2.0đ Cải tiến mở rộng)"),
        ("Công nghệ chính:", "Python 3.10/3.11, Gymnasium, PyTorch, Dueling DDQN, PPO"),
        ("Ngày lập kế hoạch:", "Tháng 11 Năm 2026")
    ]
    
    for row_idx, (k, v) in enumerate(meta_data):
        c0 = cover_tbl.cell(row_idx, 0)
        c1 = cover_tbl.cell(row_idx, 1)
        set_cell_background(c0, "F9FAFB")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=80, bottom=80, left=100, right=100)
        set_cell_margins(c1, top=80, bottom=80, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(11.5)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(v)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11.5)

    doc.add_page_break()

    # =========================================================================
    # PHẦN 1: TỔNG QUAN DỰ ÁN VÀ PHẠM VI
    # =========================================================================
    add_h1("1. TỔNG QUAN DỰ ÁN VÀ PHẠM VI CÔNG VIỆC")
    add_p(
        "Trò chơi Super Mario Bros (NES, 1985) là một trong những môi trường benchmark kinh điển nhưng đầy thử thách "
        "trong lĩnh vực Deep Reinforcement Learning (DRL). Khác với các trò chơi Atari cổ điển (Pong, Breakout) có màn hình cố định, "
        "Super Mario Bros là trò chơi cuộn cảnh màn hình ngang (side-scrolling platformer) với cơ chế vật lý phức tạp, "
        "đòi hỏi tác nhân AI phải học cách phối hợp các chuỗi hành động nhạy bén theo thời gian thực (vừa chạy lấy đà, vừa nhảy xa, "
        "né tránh quái vật Goomba/Koopa Troopa và vượt qua các hố sâu tử thần).",
        bold_prefix="Bối cảnh học thuật: "
    )
    add_p(
        "Dự án được thực hiện bởi nhóm 4 sinh viên trong thời gian đúng 5 tuần (35 ngày làm việc). Toàn bộ kiến trúc "
        "và quy trình triển khai được chuẩn hóa để hướng tới điểm số tuyệt đối 10.0/10.0, được phân chia theo cơ cấu yêu cầu:",
        bold_prefix="Quy mô và Mục tiêu điểm số: "
    )
    add_bullet(
        "Xây dựng thành công một tác nhân AI sử dụng Deep Reinforcement Learning điều khiển Mario vượt qua trọn vẹn màn chơi World 1-1, "
        "chạm cột cờ chiến thắng (flag_get == True), đạt tỷ lệ thắng (Win Rate) >= 80% trên tập kiểm thử độc lập gồm 20 seeds ngẫu nhiên. "
        "Có chương trình chạy demo trực quan, mã nguồn sạch sẽ, báo cáo khoa học hoàn chỉnh và slide thuyết trình chuyên nghiệp.",
        bold_prefix="Phần 1 - Mục tiêu cốt lõi (8.0 điểm): "
    )
    add_bullet(
        "Triển khai các nghiên cứu học thuật mở rộng có giá trị khoa học thực chất: (1) Nghiên cứu triệt tiêu (Ablation Study) về Định hình phần thưởng (Reward Shaping) "
        "giải quyết các bẫy hành vi kẹt ống nước; (2) So sánh đối kháng hiệu năng giữa thuật toán Dueling Double DQN (Off-Policy) và PPO (On-Policy); "
        "(3) Xây dựng giao diện hiển thị Heads-up Display (HUD) thời gian thực và công cụ tự động xuất video benchmark MP4 chất lượng cao.",
        bold_prefix="Phần 2 - Cải tiến & Mở rộng (2.0 điểm): "
    )
    
    add_callout(
        "Nguyên tắc sống còn của dự án: Toàn bộ nguồn lực kỹ thuật trong 3 tuần đầu tiên phải tập trung 100% cho Mục tiêu chính 8.0 điểm. "
        "Tuyệt đối không phân tán tài nguyên sang phần mở rộng 2.0 điểm khi tác nhân chưa chạm được cột cờ World 1-1 một cách ổn định.",
        "NGUYÊN TẮC QUẢN TRỊ ƯU TIÊN"
    )

    # =========================================================================
    # PHẦN 2: CHIẾN LƯỢC VÀ PHƯƠNG PHÁP TRIỂN KHAI
    # =========================================================================
    add_h1("2. CHIẾN LƯỢC VÀ PHƯƠNG PHÁP TRIỂN KHAI KỸ THUẬT")
    add_p(
        "Hệ thống điều khiển Mario được mô hình hóa toán học chặt chẽ dưới dạng Quá trình Quyết định Markov (Markov Decision Process - MDP) "
        "với bộ 5 thành phần (S, A, P, R, gamma):",
        bold_prefix="Mô hình hóa MDP: "
    )
    add_bullet(
        "Khung hình gốc của máy NES là ảnh màu 240x256x3 (RGB). Để tối ưu hóa tài nguyên tính toán và bộ nhớ RAM/VRAM, "
        "ảnh được chuyển đổi sang thang độ xám (Grayscale), giảm kích thước về 84x84 pixels. Sau đó, áp dụng kỹ thuật Frame Skip = 4 "
        "(tác nhân lặp lại hành động trong 4 frames và chỉ tính toán ở frame thứ 4) kết hợp Frame Stacking = 4 (ghép 4 frame xám liên tiếp). "
        "Trạng thái đầu vào mạng nơ-ron là một tensor kích thước (4, 84, 84). Việc xếp chồng khung hình giúp mạng nhận thức được "
        "vận tốc, gia tốc và hướng chuyển động của Mario và quái vật.",
        bold_prefix="Không gian trạng thái (State Space - S): "
    )
    add_bullet(
        "Không gian phím bấm gốc của NES gồm 8 nút (256 tổ hợp) được rút gọn về tập 7 hành động rời rạc (SIMPLE_MOVEMENT): "
        "['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'A', 'Left']. Không gian này bảo toàn 100% khả năng di chuyển và nhảy xa "
        "cần thiết để hoàn thành màn chơi nhưng giảm độ phức tạp tìm kiếm chính sách đi 36 lần.",
        bold_prefix="Không gian hành động (Action Space - A): "
    )
    add_bullet(
        "Kết hợp hàm thế năng tiến độ di chuyển trục X, thưởng kích hoạt cờ chiến thắng (+500), phạt thời gian trôi qua (-0.1), "
        "phạt mất mạng (-50) và cơ chế phạt dậm chân tại chỗ (Anti-stuck Penalty) để triệt tiêu bẫy kẹt ống nước.",
        bold_prefix="Hàm phần thưởng định hình (Shaped Reward Function - R): "
    )
    add_bullet(
        "Thiết lập gamma = 0.99, đảm bảo tác nhân có tầm nhìn dài hạn hướng tới đích đến cuối màn thay vì chỉ tối ưu hóa các phần thưởng ngắn hạn.",
        bold_prefix="Hệ số chiết khấu (Discount Factor - gamma): "
    )

    add_h2("Lựa chọn Thuật toán chính & Thuật toán đối sánh")
    add_p(
        "Sau khi phân tích kỹ lưỡng điều kiện phần cứng máy tính sinh viên và thời hạn 5 tuần, nhóm quyết định lựa chọn:",
        bold_prefix="Chiến lược thuật toán: "
    )
    add_bullet(
        "Khắc phục triệt để hiện tượng thổi phồng giá trị Q (Overestimation Bias) của Standard DQN bằng cách tách rời việc chọn hành động "
        "(Online Network) và đánh giá giá trị (Target Network). Đồng thời, kiến trúc Dueling phân rã Q(s, a) thành Giá trị trạng thái V(s) "
        "và Ưu thế hành động A(s, a), giúp mạng học nhanh trạng thái nào là an toàn mà không cần thử nghiệm toàn bộ hành động.",
        bold_prefix="Thuật toán chính (Mục tiêu 8 điểm): Dueling Double DQN (Dueling DDQN) - "
    )
    add_bullet(
        "Đại diện tiêu biểu cho trường phái Policy Gradient dạng Actor-Critic với hàm mục tiêu Clipped Surrogate Objective. "
        "PPO được huấn luyện song song làm đối chứng so sánh tốc độ hội tụ và độ ổn định chính sách.",
        bold_prefix="Thuật toán mở rộng (Cải tiến 2 điểm): Proximal Policy Optimization (PPO) - "
    )

    # =========================================================================
    # PHẦN 3: PHÂN CHIA VAI TRÒ 4 THÀNH VIÊN
    # =========================================================================
    add_h1("3. PHÂN CHIA VAI TRÒ VÀ TRÁCH NHIỆM 4 THÀNH VIÊN")
    add_p(
        "Nhóm 4 thành viên được cơ cấu theo mô hình kỹ thuật cân bằng (Balanced Engineering Matrix). Mỗi thành viên là chủ sở hữu (Owner) "
        "của một trụ cột kỹ thuật chuyên môn sâu, đồng thời đóng góp trực tiếp vào mã nguồn, thử nghiệm và công tác báo cáo khoa học. "
        "Tuyệt đối không để xảy ra tình trạng có thành viên chỉ làm slide/báo cáo hoặc đứng ngoài quy trình kỹ thuật.",
        bold_prefix="Nguyên tắc phân vai: "
    )

    role_table = doc.add_table(rows=5, cols=3)
    role_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    role_table.autofit = False
    role_table.columns[0].width = Cm(3.5)
    role_table.columns[1].width = Cm(4.5)
    role_table.columns[2].width = Cm(8.0)
    set_table_borders(role_table)

    headers = ["Thành viên", "Vai trò chuyên trách", "Phạm vi sở hữu & Trách nhiệm kỹ thuật"]
    for i, h in enumerate(headers):
        c = role_table.cell(0, i)
        set_cell_background(c, "003366")
        set_cell_margins(c, top=140, bottom=140, left=120, right=120)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.name = 'Times New Roman'

    roles_data = [
        ("Người 1", "Kỹ sư Môi trường & Tiền xử lý dữ liệu (Environment Lead)", 
         "- Thiết lập môi trường nes-py tương thích Gymnasium 1.0+ trên Windows/Linux.\n"
         "- Xây dựng các Custom Wrappers: Grayscale, Resize 84x84, FrameSkip 4, FrameStack 4.\n"
         "- Đóng gói hàm make_mario_env() và script kiểm thử baseline đo FPS, RAM.\n"
         "- Soạn thảo Chương 1, 3, 9 Báo cáo học thuật."),
        ("Người 2", "Kỹ sư Thuật toán DRL & Kiến trúc Mạng (Algorithm Lead)", 
         "- Thiết kế mạng tích chập Nature CNN kết hợp Dueling Head bằng PyTorch.\n"
         "- Cài đặt thuật toán Dueling Double DQN, Huber Loss, gradient clipping.\n"
         "- Lập trình cấu trúc dữ liệu Replay Buffer (100k transitions, lưu mảng uint8).\n"
         "- Triển khai thuật toán đối sánh PPO phục vụ điểm cộng.\n"
         "- Soạn thảo Chương 2, 4, 8 Báo cáo học thuật."),
        ("Người 3", "Kỹ sư Huấn luyện, Phần thưởng & Thực nghiệm (Training Lead)", 
         "- Thiết kế và tối ưu hàm phần thưởng Reward Shaping v1 và v2 (Anti-stuck penalty).\n"
         "- Thiết lập giám sát TensorBoard thời gian thực (Reward, Loss, Epsilon, Max X-pos).\n"
         "- Trực tiếp quản lý các đợt huấn luyện 1.0M - 3.0M steps, tinh chỉnh siêu tham số.\n"
         "- Chủ trì các thí nghiệm Ablation Study và xuất dữ liệu định lượng.\n"
         "- Soạn thảo Chương 5, 6, 7 Báo cáo học thuật."),
        ("Người 4", "Kỹ sư Tích hợp, QA, Trực quan hóa & Báo cáo (QA & Academic Lead)", 
         "- Quản trị Git Flow, cấu trúc Repository, xử lý xung đột mã nguồn (Merge Conflict).\n"
         "- Lập trình bộ kiểm thử Pytest và module đánh giá định lượng độc lập evaluate.py (20 seeds).\n"
         "- Lập trình ứng dụng demo.py hiển thị HUD thông số và tự động quay video MP4.\n"
         "- Kiểm thử tính tái lập (Reproducibility) trên môi trường máy tính trắng.\n"
         "- Tổng tài biên tập Báo cáo (Chương 10, format IEEE) và điều phối Slide, Mock Defense.")
    ]

    for row_idx, (m, r, desc) in enumerate(roles_data, start=1):
        c0 = role_table.cell(row_idx, 0)
        c1 = role_table.cell(row_idx, 1)
        c2 = role_table.cell(row_idx, 2)
        bg = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        set_cell_background(c0, bg)
        set_cell_background(c1, bg)
        set_cell_background(c2, bg)
        set_cell_margins(c0, top=100, bottom=100, left=100, right=100)
        set_cell_margins(c1, top=100, bottom=100, left=100, right=100)
        set_cell_margins(c2, top=100, bottom=100, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.add_run(m).font.bold = True
        p1 = c1.paragraphs[0]
        p1.add_run(r).font.bold = True
        p2 = c2.paragraphs[0]
        p2.add_run(desc)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # =========================================================================
    # PHẦN 4: BẢNG PHÂN CÔNG TỔNG HỢP (WBS) VÀ MA TRẬN RACI
    # =========================================================================
    add_h1("4. BẢNG PHÂN CÔNG CÔNG VIỆC TỔNG HỢP (WBS) & MA TRẬN RACI")
    add_p(
        "Bảng cơ cấu phân chia công việc (Work Breakdown Structure - WBS) được phân loại theo 3 cấp độ ưu tiên: "
        "P0 (Bắt buộc cho mục tiêu 8.0 điểm), P1 (Quan trọng cho chất lượng sản phẩm), P2 (Cải tiến mở rộng 2.0 điểm).",
        bold_prefix="Quy định độ ưu tiên: "
    )

    wbs_tasks = [
        ("01", "Cài đặt môi trường ảo, tương thích Gymnasium, fix nes-py Windows/Linux", "Người 1", "Người 4", "Tuần 1 (T2-T4)", "environment.yml, doc cài đặt", "P0", "FDEDEC"),
        ("02", "Khởi tạo Git repo, quy ước branch, .gitignore, CI test sơ bộ", "Người 4", "Cả nhóm", "Tuần 1 (T2-T3)", "Repo chuẩn, README.md", "P0", "FDEDEC"),
        ("03", "Viết Custom Wrappers: Grayscale, Resize 84x84, Skip 4, Stack 4", "Người 1", "Người 2", "Tuần 1 (T4-T6)", "wrappers.py, make_env.py", "P0", "FDEDEC"),
        ("04", "Thiết kế kiến trúc Nature CNN & Dueling Q-Network (PyTorch)", "Người 2", "Người 1", "Tuần 1 (T5-T7)", "dqn_network.py, unit tests", "P0", "FDEDEC"),
        ("05", "Xây dựng script test môi trường baseline (Random agent, đo FPS, RAM)", "Người 1", "Người 3", "Tuần 1 (T7-CN)", "baseline_random.py", "P0", "FDEDEC"),
        ("06", "Lập dàn ý chi tiết Báo cáo học thuật 12 chương và phân công viết", "Người 4", "Cả nhóm", "Tuần 1 (T7-CN)", "Đề cương Báo cáo (Word)", "P1", "FEF9E7"),
        ("07", "Lập trình Replay Buffer (uint8), Huber Loss, thuật toán Double DQN", "Người 2", "Người 3", "Tuần 2 (T2-T5)", "double_dqn.py, replay.py", "P0", "FDEDEC"),
        ("08", "Thiết kế Custom Reward v1 (vận tốc X, phạt thời gian, phạt chết)", "Người 3", "Người 1", "Tuần 2 (T2-T5)", "custom_reward.py (v1)", "P0", "FDEDEC"),
        ("09", "Xây dựng pipeline huấn luyện train.py, checkpoint và TensorBoard", "Người 2", "Người 3", "Tuần 2 (T5-T7)", "train.py, logger.py", "P0", "FDEDEC"),
        ("10", "Kiểm thử tích hợp tự động, chạy thử 50k steps kiểm tra memory leak", "Người 4", "Người 2", "Tuần 2 (T7-CN)", "tests/test_pipeline.py", "P0", "FDEDEC"),
        ("11", "Soạn thảo bản thảo Chương 1, 2, 3 của Báo cáo học thuật", "Người 1, 2", "Người 4", "Tuần 2 (T6-CN)", "Bản thảo Chương 1, 2, 3", "P1", "FEF9E7"),
        ("12", "Huấn luyện chuyên sâu mô hình chính Dueling DDQN từ 1.0M đến 3.0M steps", "Người 3", "Người 2", "Tuần 3 (T2-T5)", "best_model.pth, log TensorBoard", "P0", "FDEDEC"),
        ("13", "Nâng cấp Reward v2: Khắc phục bẫy kẹt ống nước, Anti-stuck, thưởng cờ", "Người 3", "Người 1", "Tuần 3 (T4-T6)", "custom_reward.py (v2)", "P0", "FDEDEC"),
        ("14", "Lập trình module evaluate.py đánh giá độc lập trên 20 seeds ngẫu nhiên", "Người 4", "Người 3", "Tuần 3 (T5-T7)", "evaluate.py, kết quả CSV", "P0", "FDEDEC"),
        ("15", "ĐẠT MỐC MARIO CHẠM CỜ WORLD 1-1 (flag_get == True), WIN RATE >= 80%", "Cả nhóm", "Cả nhóm", "Tuần 3 (T7-CN)", "Video chạm cờ, nghiệm thu P0", "P0", "FDEDEC"),
        ("16", "Soạn thảo bản thảo Chương 4, 5, 6 của Báo cáo học thuật", "Người 2, 3", "Người 4", "Tuần 3 (T6-CN)", "Bản thảo Chương 4, 5, 6", "P1", "FEF9E7"),
        ("17", "Cải tiến mở rộng 1: Huấn luyện đối sánh thuật toán PPO vs Dueling DDQN", "Người 2", "Người 3", "Tuần 4 (T2-T5)", "ppo_agent.py, đồ thị so sánh", "P2", "E8F8F5"),
        ("18", "Cải tiến mở rộng 2: Ablation Study về Reward Shaping và Action Space", "Người 3", "Người 1", "Tuần 4 (T4-T7)", "Bảng số liệu Ablation", "P2", "E8F8F5"),
        ("19", "Lập trình demo.py hiển thị HUD thời gian thực và tự động xuất video MP4", "Người 4", "Người 1", "Tuần 4 (T5-T7)", "demo.py, video có HUD", "P1", "FEF9E7"),
        ("20", "Kiểm thử tính tái lập (Reproducibility) sạch sẽ trên máy tính thứ hai", "Người 4", "Người 1", "Tuần 4 (T7-CN)", "Báo cáo kiểm thử tái lập", "P1", "FEF9E7"),
        ("21", "Soạn thảo bản thảo Chương 7, 8, 9 của Báo cáo học thuật", "Người 3, 4", "Cả nhóm", "Tuần 4 (T6-CN)", "Bản thảo Chương 7, 8, 9", "P1", "FEF9E7"),
        ("22", "Khóa mã nguồn (Code Freeze), dọn dẹp refactor, gắn tag Release v1.0.0", "Người 4", "Cả nhóm", "Tuần 5 (T2-T3)", "Release v1.0.0 trên Git", "P0", "FDEDEC"),
        ("23", "Hoàn thiện toàn diện Báo cáo học thuật (Chương 1-12, format IEEE, PDF)", "Người 4", "Cả nhóm", "Tuần 5 (T3-T5)", "Báo cáo chính thức (.pdf, .docx)", "P0", "FDEDEC"),
        ("24", "Thiết kế Slide thuyết trình chuyên nghiệp (25-28 slides, đồ họa sắc nét)", "Người 4", "Cả nhóm", "Tuần 5 (T4-T6)", "Slide thuyết trình (.pptx)", "P0", "FDEDEC"),
        ("25", "Biên soạn Ngân hàng 20 câu hỏi và đáp án vấn đáp lý thuyết & thực hành", "Cả nhóm", "Cả nhóm", "Tuần 5 (T5-T6)", "Tài liệu ôn tập vấn đáp", "P0", "FDEDEC"),
        ("26", "Tổ chức 02 buổi diễn tập bảo vệ thử nghiệm có bấm giờ (Mock Defense)", "Cả nhóm", "Cả nhóm", "Tuần 5 (T6-T7)", "Biên bản góp ý diễn tập", "P0", "FDEDEC")
    ]

    wbs_table = doc.add_table(rows=len(wbs_tasks) + 1, cols=7)
    wbs_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    wbs_table.autofit = False
    wbs_table.columns[0].width = Cm(1.0)
    wbs_table.columns[1].width = Cm(5.2)
    wbs_table.columns[2].width = Cm(1.8)
    wbs_table.columns[3].width = Cm(1.8)
    wbs_table.columns[4].width = Cm(2.2)
    wbs_table.columns[5].width = Cm(2.8)
    wbs_table.columns[6].width = Cm(1.2)
    set_table_borders(wbs_table)

    wbs_headers = ["STT", "Hạng mục công việc (WBS)", "Chủ trì", "Hỗ trợ", "Thời hạn", "Sản phẩm bàn giao", "Ưu tiên"]
    for i, h in enumerate(wbs_headers):
        c = wbs_table.cell(0, i)
        set_cell_background(c, "003366")
        set_cell_margins(c, top=120, bottom=120, left=80, right=80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.name = 'Times New Roman'

    for row_idx, item in enumerate(wbs_tasks, start=1):
        stt, task, owner, co, time, out, pri, col = item
        for col_idx, val in enumerate([stt, task, owner, co, time, out, pri]):
            c = wbs_table.cell(row_idx, col_idx)
            set_cell_background(c, col)
            set_cell_margins(c, top=80, bottom=80, left=60, right=60)
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(0)
            if col_idx in [0, 2, 3, 4, 6]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9)
            if col_idx == 6:
                r.font.bold = True
                if pri == "P0":
                    r.font.color.rgb = RGBColor(180, 0, 0)
                elif pri == "P1":
                    r.font.color.rgb = RGBColor(150, 100, 0)
                else:
                    r.font.color.rgb = RGBColor(0, 120, 80)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_h2("Ma trận RACI (Phân định trách nhiệm tuyệt đối)")
    add_p(
        "Ma trận RACI đảm bảo mỗi nhiệm vụ kỹ thuật chỉ có duy nhất MỘT người chịu trách nhiệm giải trình cuối cùng (Accountable), "
        "loại bỏ hoàn toàn tình trạng đùn đẩy trách nhiệm hoặc chồng chéo quyền hạn.",
        bold_prefix="Nguyên tắc RACI: "
    )

    raci_table = doc.add_table(rows=11, cols=5)
    raci_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    raci_table.autofit = False
    raci_table.columns[0].width = Cm(6.0)
    raci_table.columns[1].width = Cm(2.5)
    raci_table.columns[2].width = Cm(2.5)
    raci_table.columns[3].width = Cm(2.5)
    raci_table.columns[4].width = Cm(2.5)
    set_table_borders(raci_table)

    raci_headers = ["Lĩnh vực chuyên môn / Hạng mục", "Người 1", "Người 2", "Người 3", "Người 4"]
    for i, h in enumerate(raci_headers):
        c = raci_table.cell(0, i)
        set_cell_background(c, "003366")
        set_cell_margins(c, top=100, bottom=100, left=80, right=80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    raci_data = [
        ("Môi trường Gymnasium & Custom Wrappers", "A / R", "C", "C", "I"),
        ("Kiến trúc Mạng CNN & Dueling Head", "C", "A / R", "C", "I"),
        ("Replay Buffer & Thuật toán Double DQN", "I", "A / R", "R", "I"),
        ("Reward Engineering & Khắc phục bẫy kẹt", "C", "C", "A / R", "I"),
        ("Vận hành Huấn luyện 3M steps & Tuning", "I", "C", "A / R", "I"),
        ("Cấu trúc Repository, Git Flow & CI Test", "I", "I", "I", "A / R"),
        ("Đánh giá Benchmark 20 seeds & Demo HUD", "I", "I", "C", "A / R"),
        ("Thí nghiệm Mở rộng (PPO & Ablation)", "C", "R", "R", "A"),
        ("Tổng hợp & Biên tập Báo cáo học thuật", "R (Ch.1,3,9)", "R (Ch.2,4,8)", "R (Ch.5,6,7)", "A / R (Tổng tài)"),
        ("Thiết kế Slide & Tổ chức Diễn tập", "R", "R", "R", "A / R")
    ]

    for row_idx, row in enumerate(raci_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(row):
            c = raci_table.cell(row_idx, col_idx)
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.add_run(val).font.name = 'Times New Roman'
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(val)
                r.font.name = 'Times New Roman'
                if "A" in val:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(180, 0, 0)
                elif "R" in val:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0, 102, 153)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # PHẦN 5: LỘ TRÌNH TRIỂN KHAI 5 TUẦN (TIMELINE)
    # =========================================================================
    add_h1("5. LỘ TRÌNH TRIỂN KHAI CHI TIẾT 5 TUẦN (TIMELINE)")
    
    weeks_info = [
        ("TUẦN 1: THIẾT LẬP NỀN TẢNG, MÔI TRƯỜNG & THIẾT KẾ KIẾN TRÚC (NGÀY 1 - 7)",
         "Mục tiêu cốt lõi: Cả 4 thành viên cài đặt thành công môi trường Mario trên máy cá nhân không có lỗi xung đột thư viện; cấu trúc mạng nơ-ron và quy ước Git được thống nhất hoàn toàn.",
         [
             "Ngày 1-2: Người 4 tạo Git repository, file .gitignore. Người 1 khảo sát nes-py và Gymnasium, lập environment.yml.",
             "Ngày 3-4: Người 1 hoàn thiện Grayscale, Resize 84x84, FrameSkip 4, FrameStack 4. Người 2 thiết kế DuelingDQN trong PyTorch.",
             "Ngày 5-6: Người 1 đóng gói make_mario_env() và viết script baseline_random.py đo FPS. Người 4 lập dàn ý Báo cáo.",
             "Ngày 7: Họp Sprint 1: Nghiệm thu môi trường chạy mượt mà >= 60 FPS, tensor output chuẩn (4, 84, 84)."
         ],
         "requirements.txt, wrappers.py, make_env.py, dqn_network.py, dàn ý Báo cáo.",
         "Môi trường khởi tạo mượt mà không crash sau 10,000 steps; Pytest tensor shape đạt 100%."),
        
        ("TUẦN 2: XÂY DỰNG PHIÊN BẢN THUẬT TOÁN ĐẦU TIÊN (ALPHA PIPELINE) (NGÀY 8 - 14)",
         "Mục tiêu cốt lõi: Có pipeline hoàn chỉnh chạy được từ đầu đến cuối; tác nhân bắt đầu tương tác với game và biết học cách nhấn phím tiến về bên phải; hệ thống log TensorBoard hoạt động ổn định.",
         [
             "Ngày 8-9: Người 2 lập trình ReplayBuffer (100k) và DoubleDQNAgent. Người 3 thiết kế CustomRewardWrapper v1.",
             "Ngày 10-11: Người 2 và Người 3 ghép nối pipeline thành script train.py, tích hợp TensorBoard log và checkpoint.",
             "Ngày 12-13: Người 4 viết unit test kiểm tra toàn diện luồng dữ liệu. Chạy thử nghiệm 100k steps kiểm tra rò rỉ RAM.",
             "Ngày 14: Họp Sprint 2: Phân tích đường cong học tập ban đầu. Đảm bảo Mario đã biết học cách bấm phím sang phải."
         ],
         "double_dqn.py, replay_buffer.py, custom_reward.py (v1), train.py, bản thảo Chương 1, 2, 3 Báo cáo.",
         "Chương trình huấn luyện chạy liên tục 100,000 steps không lỗi ngoại lệ; Mario vượt qua mốc tọa độ X > 300."),
        
        ("TUẦN 3: HUẤN LUYỆN CHUYÊN SÂU & HOÀN THÀNH MỤC TIÊU 8 ĐIỂM (NGÀY 15 - 21)",
         "Mục tiêu cốt lõi: MỐC SỐNG CÒN CỦA ĐỒ ÁN. Tác nhân hoàn thành xuất sắc màn chơi World 1-1 (chạm cờ flag_get == True), tỷ lệ thắng đạt >= 80% trên tập kiểm thử độc lập gồm 20 seeds ngẫu nhiên.",
         [
             "Ngày 15-16: Người 3 cấu hình siêu tham số tối ưu (lr=1e-4, batch=64, gamma=0.99), bắt đầu đợt huấn luyện lớn 3M steps.",
             "Ngày 17-18: Người 3 và Người 1 tinh chỉnh Reward v2 (Anti-stuck penalty, phạt chết nặng -50, thưởng cờ +500) để triệt tiêu bẫy kẹt ống nước.",
             "Ngày 19-20: Người 4 hoàn thiện script evaluate.py. Mô hình đạt mốc vượt qua hố sâu và chạm cờ chiến thắng (X >= 3161).",
             "Ngày 21: Họp Sprint 3: Cả nhóm xác nhận chính thức hoàn thành mục tiêu 8.0 điểm. Lưu trữ checkpoint vàng và video chiến thắng."
         ],
         "checkpoints/best_model.pth, evaluate.py, bảng kết quả 20 seeds, video chạm cờ, bản thảo Chương 4, 5, 6 Báo cáo.",
         "Win Rate >= 80% trên 20 test episodes độc lập; cờ flag_get == True được engine game xác nhận."),
        
        ("TUẦN 4: CẢI TIẾN MỞ RỘNG (2 ĐIỂM CỘNG) & HOÀN THIỆN KỸ THUẬT (NGÀY 22 - 28)",
         "Mục tiêu cốt lõi: Thực hiện thành công các nghiên cứu đối sánh học thuật có giá trị cao; lập trình giao diện demo HUD sắc nét; kiểm thử tính tái lập trên máy tính thứ hai độc lập.",
         [
             "Ngày 22-23: Người 2 triển khai thuật toán đối sánh PPO. Người 3 thiết lập thí nghiệm đối kháng DDQN vs PPO.",
             "Ngày 24-25: Người 3 triển khai Ablation Study về Reward v1 vs v2 và Không gian hành động (SIMPLE vs RIGHT_ONLY).",
             "Ngày 26-27: Người 4 lập trình demo.py hiển thị HUD và ghi hình MP4. Tiến hành test tái lập sạch trên máy tính trắng.",
             "Ngày 28: Họp Sprint 4: Nghiệm thu toàn bộ dữ liệu thực nghiệm phần mở rộng 2 điểm. ĐÓNG BĂNG MÃ NGUỒN (CODE FREEZE)."
         ],
         "ppo_agent.py, bảng số liệu Ablation Study, demo.py, video MP4 chất lượng cao, bản thảo Chương 7, 8, 9 Báo cáo.",
         "Có số liệu so sánh định lượng của ít nhất 2 nghiên cứu mở rộng; demo.py chạy mượt mà có thanh hiển thị HUD."),
        
        ("TUẦN 5: KIỂM THỬ CUỐI, BÁO CÁO, SLIDE & TỔ CHỨC DIỄN TẬP VẤN ĐÁP (NGÀY 29 - 35)",
         "Mục tiêu cốt lõi: Đóng gói trọn vẹn sản phẩm đồ án; xuất bản Báo cáo học thuật chuẩn IEEE; Slide thuyết trình hoàn hảo; cả 4 thành viên thuộc bài và trả lời lưu loát mọi câu hỏi của hội đồng.",
         [
             "Ngày 29-30: Người 4 gắn thẻ Release v1.0.0 trên GitHub. Hoàn thiện toàn diện Báo cáo học thuật 12 chương dạng PDF/Word.",
             "Ngày 31-32: Cả nhóm rà soát lỗi chính tả, công thức toán học. Người 4 hoàn thiện bộ Slide thuyết trình 25-28 trang.",
             "Ngày 33: Cả nhóm hoàn thành tài liệu Ngân hàng 20 câu hỏi và đáp án vấn đáp, ôn tập kỹ càng.",
             "Ngày 34: Tổ chức Buổi diễn tập bảo vệ thử nghiệm lần 1 (Mock Defense 1) - Bấm giờ chính xác 20 phút thuyết trình.",
             "Ngày 35: Tổ chức Buổi diễn tập bảo vệ thử nghiệm lần 2 (Mock Defense 2) - Phản biện câu hỏi hóc búa. SẴN SÀNG BẢO VỆ CHÍNH THỨC."
         ],
         "Báo cáo học thuật hoàn chỉnh (PDF/Word), Slide thuyết trình (.pptx), Release v1.0.0, Bộ câu hỏi vấn đáp, Thư mục demo offline.",
         "Báo cáo đạt chuẩn định dạng hình thức; 4 thành viên thuyết trình khớp thời lượng 20 phút và trả lời trôi chảy 100% câu hỏi.")
    ]

    for title, goal, actions, outs, criteria in weeks_info:
        add_h2(title)
        add_p(goal, bold_prefix="Mục tiêu trọng tâm: ")
        add_p("Các hành động triển khai cụ thể theo ngày:", bold_prefix="Kế hoạch tác chiến: ")
        for act in actions:
            add_bullet(act)
        add_p(outs, bold_prefix="Sản phẩm đầu ra bắt buộc: ")
        add_p(criteria, bold_prefix="Tiêu chí nghiệm thu (DoD): ")
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # =========================================================================
    # PHẦN 6-9: SỔ TAY HÀNH ĐỘNG CHI TIẾT RIÊNG CHO TỪNG THÀNH VIÊN
    # =========================================================================
    add_h1("6. QUY TRÌNH THỰC HIỆN CHI TIẾT - NGƯỜI 1 (MÔI TRƯỜNG & TIỀN XỬ LÝ)")
    add_p(
        "Người 1 là 'người gác cổng' của toàn bộ pipeline dữ liệu. Nếu môi trường bị lỗi, sai lệch kích thước tensor, "
        "hoặc bị rò rỉ bộ nhớ thì toàn bộ quá trình huấn luyện của Người 2 và Người 3 sẽ sụp đổ.",
        bold_prefix="Tầm quan trọng sống còn: "
    )
    add_h2("Trình tự hành động kỹ thuật của Người 1:")
    add_bullet("Khảo sát nes-py v8.2.1 và gym-super-mario-bros v7.4.0. Khuyến nghị toàn nhóm sử dụng Python 3.10 hoặc 3.11 trong môi trường Conda chuẩn để có sẵn pre-built binary wheel, tránh lỗi thiếu bộ biên dịch C++ trên Windows.", bold_prefix="Bước 1 (Thiết lập môi trường không lỗi): ")
    add_bullet("Viết lớp GymnasiumCompatibilityWrapper để bọc hàm step() trả về đúng 5 giá trị: (obs, reward, terminated, truncated, info), ngăn chặn hoàn toàn lỗi vỡ code 'too many values to unpack'.", bold_prefix="Bước 2 (Chuẩn hóa API Gymnasium): ")
    add_bullet("Lập trình GrayScaleAndResizeWrapper chuyển ảnh 240x256x3 sang ảnh xám 84x84x1 bằng cv2.INTER_AREA (giảm 96% kích thước dữ liệu mà không làm mất đặc trưng chướng ngại vật).", bold_prefix="Bước 3 (Thu nhỏ kích thước ảnh): ")
    add_bullet("Lập trình CustomFrameSkipAndStackWrapper: Bỏ qua 4 khung hình (FrameSkip=4), lấy max giữa 2 frame cuối để loại bỏ hiện tượng nhấp nháy hình (NES Sprite Flickering), và xếp chồng 4 frame xám thành tensor (4, 84, 84).", bold_prefix="Bước 4 (Khôi phục tính chất Markov bằng Frame Stacking): ")
    add_bullet("Rút gọn từ 256 phím NES về 7 hành động tối ưu SIMPLE_MOVEMENT thông qua JoypadSpace.", bold_prefix="Bước 5 (Tối ưu hóa không gian hành động): ")
    add_bullet("Đóng gói toàn bộ thành hàm make_mario_env(world=1, stage=1, action_type='SIMPLE') và viết script baseline_random.py đo FPS.", bold_prefix="Bước 6 (Bàn giao hàm chuẩn hóa): ")

    add_h1("7. QUY TRÌNH THỰC HIỆN CHI TIẾT - NGƯỜI 2 (THUẬT TOÁN HỌC TĂNG CƯỜNG)")
    add_p(
        "Người 2 chịu trách nhiệm 'bộ não' của hệ thống. Lập trình kiến trúc mạng nơ-ron tích chập và cài đặt thuật toán "
        "Dueling Double DQN đạt độ chính xác toán học tuyệt đối.",
        bold_prefix="Tầm quan trọng sống còn: "
    )
    add_h2("Trình tự hành động kỹ thuật của Người 2:")
    add_bullet("Lập trình class DuelingDQN trong PyTorch gồm 3 tầng tích chập trích xuất đặc trưng (Conv1: 32 filters 8x8 stride 4; Conv2: 64 filters 4x4 stride 2; Conv3: 64 filters 3x3 stride 1; Flatten ra 3136 chiều).", bold_prefix="Bước 1 (Xây dựng Feature Extractor Backbone): ")
    add_bullet("Tách thành 2 luồng Fully Connected: Luồng Giá trị V(s) (Linear 512 -> 1) và Luồng Ưu thế A(s, a) (Linear 512 -> 7). Ghép lại theo công thức chuẩn hóa trung bình: Q = V + (A - mean(A)).", bold_prefix="Bước 2 (Kiến trúc Dueling Head): ")
    add_bullet("Lập trình ReplayBuffer lưu trữ 100,000 transitions dưới dạng mảng Numpy uint8 trên RAM hệ thống (chỉ chiếm ~2.8 GB RAM). Chỉ convert sang PyTorch Tensor trên GPU khi hàm sample(batch_size=64) được gọi.", bold_prefix="Bước 3 (Bộ nhớ kinh nghiệm chống tràn RAM GPU): ")
    add_bullet("Cài đặt Double DQN target: a* = argmax Q(s', a; theta_online), Target = R + gamma * Q(s', a*; theta_target). Sử dụng Huber Loss (SmoothL1Loss) và gradient clipping max_norm=10.0.", bold_prefix="Bước 4 (Triển khai Double Target & Loss Huber): ")
    add_bullet("Cài đặt cơ chế cập nhật Mạng đích (Target Network hard update) mỗi 10,000 training steps một lần.", bold_prefix="Bước 5 (Đồng bộ hóa Target Network): ")
    add_bullet("Cài đặt thuật toán Actor-Critic PPO hoặc thiết lập pipeline đối sánh bằng Stable-Baselines3 làm đối chứng cho phần mở rộng 2 điểm.", bold_prefix="Bước 6 (Mô hình đối sánh PPO): ")

    add_h1("8. QUY TRÌNH THỰC HIỆN CHI TIẾT - NGƯỜI 3 (HUẤN LUYỆN, PHẦN THƯỞNG & THỰC NGHIỆM)")
    add_p(
        "Người 3 là người trực tiếp 'dạy dỗ' Mario. Dù thuật toán có hoàn hảo đến đâu, nếu hàm phần thưởng thiết kế sai lệch, "
        "tác nhân sẽ tìm ra các kẽ hở hành vi kỳ dị (đứng yên, nhảy lùi, dậm chân tại chỗ) thay vì về đích.",
        bold_prefix="Tầm quan trọng sống còn: "
    )
    add_h2("Trình tự hành động kỹ thuật của Người 3:")
    add_bullet("Phân tích 4 bẫy hành vi: Bẫy sợ chết nhảy lùi, bẫy kẹt ống nước cao, bẫy rơi hố sâu và bẫy nhảy tại chỗ câu giờ.", bold_prefix="Bước 1 (Nhận diện các bẫy tâm lý của Mario): ")
    add_bullet("Lập trình MarioCustomRewardWrapper: thưởng tiến độ trục X (delta_x * 0.5), thưởng phá kỷ lục X_max, phạt thời gian (-0.1/step), phạt chết nặng (-50), và THƯỞNG CỰC LỚN khi chạm cờ (+500).", bold_prefix="Bước 2 (Thiết kế Reward Shaping v2): ")
    add_bullet("Tích hợp cơ chế Anti-stuck: Nếu tọa độ X không tăng thêm quá 2 pixel trong 50 bước liên tiếp -> phạt điểm âm lũy tiến; nếu kẹt quá 200 bước -> cưỡng chế ngắt episode để học khám phá mới.", bold_prefix="Bước 3 (Cơ chế chống kẹt ống nước đột phá): ")
    add_bullet("Cấu hình TensorBoard theo dõi 5 chỉ số vàng: Episode Reward, Max X-Position, Episode Length, Huber Loss, Epsilon.", bold_prefix="Bước 4 (Thiết lập giám sát thời gian thực): ")
    add_bullet("Vận hành đợt huấn luyện lớn 3.0 triệu steps. Giám sát đường cong hội tụ, lưu checkpoint định kỳ mỗi 50k steps và sao lưu best_model.pth lên Google Drive.", bold_prefix="Bước 5 (Huấn luyện mốc 8 điểm): ")
    add_bullet("Chủ trì 2 thí nghiệm Ablation Study: (1) So sánh Reward v1 vs v2; (2) So sánh không gian hành động SIMPLE vs RIGHT_ONLY. Xuất dữ liệu biểu đồ độ phân giải cao phục vụ báo cáo.", bold_prefix="Bước 6 (Thực nghiệm lấy trọn 2 điểm mở rộng): ")

    add_h1("9. QUY TRÌNH THỰC HIỆN CHI TIẾT - NGƯỜI 4 (TÍCH HỢP, KIỂM THỬ, ĐÁNH GIÁ & BÁO CÁO)")
    add_p(
        "Người 4 giữ vai trò 'tổng công trình sư' về chất lượng sản phẩm và điều phối học thuật. Người 4 đóng góp trực tiếp "
        "bằng mã nguồn kiểm thử định lượng, công cụ demo HUD, đồng thời dẫn dắt nhóm hoàn thiện Báo cáo và Slide đạt chuẩn xuất sắc.",
        bold_prefix="Tầm quan trọng sống còn: "
    )
    add_h2("Trình tự hành động kỹ thuật của Người 4:")
    add_bullet("Thiết lập Git repository, file .gitignore, cấu trúc thư mục module chuẩn quốc tế và quy định quy trình PR review trước khi merge.", bold_prefix="Bước 1 (Quản trị Repository & CI/CD): ")
    add_bullet("Lập trình bộ kiểm thử tự động tests/test_env.py và tests/test_pipeline.py kiểm tra kích thước tensor và khả năng giải phóng RAM.", bold_prefix="Bước 2 (Xây dựng Pytest Suite): ")
    add_bullet("Lập trình evaluate.py: Tắt hoàn toàn khám phá ngẫu nhiên (epsilon=0.0), chạy kiểm thử khách quan trên 20 seeds ngẫu nhiên cố định, xuất ma trận kết quả (Win Rate, Max X, Steps) ra file CSV.", bold_prefix="Bước 3 (Module đánh giá định lượng khoa học): ")
    add_bullet("Lập trình demo.py: Render khung hình RGB NES sắc nét 30 FPS, vẽ thanh thông số HUD trực quan (Action, Max Q-value, Pos X, Step) và tự động ghi video MP4 full-HD.", bold_prefix="Bước 4 (Ứng dụng Demo thời gian thực kèm HUD): ")
    add_bullet("Thực hiện kiểm thử tính tái lập (Reproducibility): Clone repo sang máy tính mới hoàn toàn, chạy lại thành công toàn bộ pipeline theo file README.md.", bold_prefix="Bước 5 (Kiểm thử tái lập trên máy trắng): ")
    add_bullet("Tổng tài biên tập Báo cáo học thuật 12 chương (chuẩn font Times New Roman, trích dẫn IEEE), hoàn thiện Slide 25-28 trang và điều phối 2 buổi Mock Defense bấm giờ.", bold_prefix="Bước 6 (Điều phối Báo cáo, Slide và Vấn đáp): ")

    # =========================================================================
    # PHẦN 10: QUY TRÌNH PHỐI HỢP VÀ QUẢN LÝ MÃ NGUỒN
    # =========================================================================
    add_h1("10. QUY TRÌNH PHỐI HỢP NHÓM VÀ QUẢN LÝ MÃ NGUỒN")
    add_bullet("Nhánh main chỉ chứa các phiên bản ổn định tuyệt đối đã qua kiểm thử (Release). Nhánh develop là nhánh tích hợp chung. Mỗi thành viên làm việc trên nhánh tính năng riêng: feature/env-wrappers (Người 1), feature/dueling-dqn (Người 2), feature/reward-shaping (Người 3), feature/eval-and-demo (Người 4).", bold_prefix="Chiến lược phân nhánh Git Flow: ")
    add_bullet("Mỗi Pull Request bắt buộc phải có ít nhất 01 thành viên khác review và approve. Người 4 là người duyệt cuối cùng để tránh xung đột mã nguồn.", bold_prefix="Quy định Pull Request & Code Review: ")
    add_bullet("Áp dụng chuẩn Conventional Commits: feat: (tính năng mới), fix: (sửa lỗi), docs: (tài liệu/báo cáo), test: (kiểm thử), refactor: (tái cấu trúc mã nguồn).", bold_prefix="Quy ước Commit Message: ")
    add_bullet("File trọng số mô hình lớn (.pth) và log Tensorboard không commit trực tiếp lên Git để tránh làm phình dung lượng repo. Toàn bộ trọng số được lưu tại checkpoints/ và tự động đồng bộ lên Google Drive của nhóm.", bold_prefix="Quản lý File lớn & Checkpoints: ")
    add_bullet("Họp Daily Standup 10 phút (Thứ 2, 4, 6 lúc 21h00 qua Discord/Google Meet). Họp Sprint Review 60 phút vào tối Chủ Nhật hàng tuần để nghiệm thu và lập kế hoạch tuần mới.", bold_prefix="Lịch họp và Giao tiếp nội bộ: ")

    # =========================================================================
    # PHẦN 11: KẾ HOẠCH BÁO CÁO HỌC THUẬT (12 CHƯƠNG)
    # =========================================================================
    add_h1("11. KẾ HOẠCH THỰC HIỆN BÁO CÁO HỌC THUẬT (12 CHƯƠNG)")
    add_p(
        "Báo cáo học thuật cuối kỳ là sản phẩm chiếm tỷ trọng điểm số lớn. Kế hoạch biên soạn được thực hiện theo nguyên tắc "
        "cuốn chiếu liên tục trong 5 tuần, phân công rành mạch tác giả chính cho từng chương:",
        bold_prefix="Nguyên tắc biên tập: "
    )

    report_table = doc.add_table(rows=13, cols=4)
    report_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    report_table.autofit = False
    report_table.columns[0].width = Cm(2.0)
    report_table.columns[1].width = Cm(6.5)
    report_table.columns[2].width = Cm(4.5)
    report_table.columns[3].width = Cm(3.0)
    set_table_borders(report_table)

    rep_headers = ["Chương", "Tên chương & Nội dung trọng tâm", "Tác giả chính / Hỗ trợ", "Hạn nộp bản thảo"]
    for i, h in enumerate(rep_headers):
        c = report_table.cell(0, i)
        set_cell_background(c, "003366")
        set_cell_margins(c, top=100, bottom=100, left=80, right=80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rep_data = [
        ("Mở đầu", "Abstract, Lời cảm ơn, Danh mục hình vẽ & bảng biểu", "Người 4 (Chủ trì)", "Tuần 5 (Ngày 31)"),
        ("Chương 1", "Giới thiệu đề tài, Bối cảnh game Mario, Mục tiêu 8đ & 2đ", "Người 1 / Người 4", "Tuần 2 (Ngày 14)"),
        ("Chương 2", "Cơ sở lý thuyết RL, MDP, Bellman, Double DQN, Dueling", "Người 2 / Người 3", "Tuần 2 (Ngày 14)"),
        ("Chương 3", "Phân tích bài toán Mario: Grayscale, Resize, Skip 4, Stack 4", "Người 1 / Người 2", "Tuần 2 (Ngày 14)"),
        ("Chương 4", "Thiết kế hệ thống: CNN Backbone, Luồng Value/Advantage", "Người 2 / Người 1", "Tuần 3 (Ngày 21)"),
        ("Chương 5", "Reward Shaping: Phân tích 4 bẫy hành vi, công thức v1/v2", "Người 3 / Người 2", "Tuần 3 (Ngày 21)"),
        ("Chương 6", "Quy trình thực nghiệm: Siêu tham số, Phần cứng, TensorBoard", "Người 3 / Người 4", "Tuần 3 (Ngày 21)"),
        ("Chương 7", "Kết quả thực nghiệm giải màn 1-1 (20 seeds, Win Rate 85%)", "Người 3 / Người 4", "Tuần 4 (Ngày 28)"),
        ("Chương 8", "Nghiên cứu mở rộng: Đối sánh PPO vs DDQN, Ablation Study", "Người 2, 3 / Người 4", "Tuần 4 (Ngày 28)"),
        ("Chương 9", "Đánh giá hạn chế, Hiện tượng quá khớp & Hướng mở rộng", "Người 1 / Người 3", "Tuần 4 (Ngày 28)"),
        ("Chương 10", "Hướng dẫn cài đặt, Kiểm thử tái lập (Reproducibility Guide)", "Người 4 / Người 1", "Tuần 5 (Ngày 30)"),
        ("Kết luận", "Tổng kết đóng góp kỹ thuật & Danh mục 15 tài liệu IEEE", "Người 4 / Cả nhóm", "Tuần 5 (Ngày 31)")
    ]

    for row_idx, (ch, name, auth, dead) in enumerate(rep_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate([ch, name, auth, dead]):
            c = report_table.cell(row_idx, col_idx)
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if col_idx in [0, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9)
            if col_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # PHẦN 12: KẾ HOẠCH SLIDE VÀ THUYẾT TRÌNH
    # =========================================================================
    add_h1("12. KẾ HOẠCH XÂY DỰNG SLIDE VÀ PHÂN CHIA THUYẾT TRÌNH")
    add_p(
        "Bộ slide được thiết kế chuyên nghiệp trên Canva/PowerPoint gồm 25-28 slides, tuân thủ nguyên tắc trực quan: "
        "Nhiều sơ đồ khối, nhiều biểu đồ TensorBoard và video minh họa, giảm tối đa chữ viết văn xuôi.",
        bold_prefix="Định dạng Slide: "
    )
    add_bullet("Slide 1-3: Giới thiệu đề tài, thách thức của Super Mario Bros và mục tiêu kép (8 điểm + 2 điểm).", bold_prefix="Khung mở đầu (Người 4 điều phối): ")
    add_bullet("Slide 4-7: Mô hình hóa MDP, giải thích trực quan tại sao cần Grayscale, Resize 84x84, FrameSkip 4 và FrameStack 4, tối ưu hóa JoypadSpace.", bold_prefix="Phần 1 - Môi trường & Tiền xử lý (Người 1 trình bày - 5 phút): ")
    add_bullet("Slide 8-12: Tại sao chọn Dueling Double DQN? Cơ chế toán học giải quyết Overestimation Bias, sơ đồ phân tách luồng Value/Advantage và Replay Buffer.", bold_prefix="Phần 2 - Thuật toán & Mạng nơ-ron (Người 2 trình bày - 5 phút): ")
    add_bullet("Slide 13-17: Giải phẫu 4 bẫy hành vi kẹt ống nước; công thức Reward Shaping v2; đường cong học tập TensorBoard; các thí nghiệm Ablation Study đối sánh PPO.", bold_prefix="Phần 3 - Huấn luyện & Thực nghiệm (Người 3 trình bày - 5 phút): ")
    add_bullet("Slide 18-24: Kết quả đánh giá 20 seeds (Win Rate 85%); TRÌNH CHIẾU VIDEO DEMO MARIO CHẠM CỜ KÈM THANH HUD; kiểm thử tính tái lập; kết luận và định hướng tương lai.", bold_prefix="Phần 4 - Đánh giá, Demo & Kết luận (Người 4 trình bày - 5 phút): ")
    add_bullet("Slide 25-28: Lời cảm ơn và chuyển sang phiên hỏi đáp phản biện với Hội đồng.", bold_prefix="Khung kết thúc & Vấn đáp (Cả nhóm): ")

    # =========================================================================
    # PHẦN 13: NGÂN HÀNG CÂU HỎI VÀ ĐÁP ÁN VẤN ĐÁP CHUYÊN SÂU
    # =========================================================================
    add_h1("13. BỘ CÂU HỎI VÀ ĐÁP ÁN ÔN TẬP VẤN ĐÁP BẢO VỆ ĐỒ ÁN")
    add_p(
        "Dưới đây là 6 câu hỏi cốt lõi có tần suất xuất hiện cao nhất trong các buổi bảo vệ đồ án DRL, được soạn thảo đáp án "
        "chuẩn mực, ngắn gọn, chuẩn xác bản chất toán học và thực nghiệm:",
        bold_prefix="Ngân hàng câu hỏi: "
    )

    qa_list = [
        ("Câu 1: Tại sao nhóm phải xếp chồng 4 khung hình (Frame Stacking) thay vì chỉ đưa 1 khung hình xám vào mạng nơ-ron?",
         "Trò chơi Mario là một môi trường quan sát một phần nếu chỉ nhìn 1 khung ảnh tĩnh (POMDP). Từ một bức ảnh duy nhất, "
         "tác nhân không thể biết được Mario đang đứng yên, chạy sang phải hay nhảy sang trái; cũng không biết được con Goomba đang di chuyển "
         "về phía nào hay đạn pháo đang bay với vận tốc bao nhiêu. Bằng cách ghép 4 khung hình liên tiếp thành tensor (4, 84, 84), "
         "mạng CNN có thể nắm bắt được vận tốc, gia tốc và chiều chuyển động của mọi vật thể, đưa bài toán trở về đúng tính chất Markov đầy đủ (Fully Observable MDP)."),
        
        ("Câu 2: Hiện tượng Overestimation Bias trong Standard DQN là gì và Double DQN giải quyết nó bằng cách nào?",
         "Trong Standard DQN, mục tiêu Bellman Target dùng toán tử max trên cùng bộ trọng số Target Network: Y = R + gamma * max_a Q(s', a; theta^-). "
         "Do hàm xấp xỉ mạng nơ-ron luôn có sai số ngẫu nhiên, toán tử max sẽ có xu hướng luôn chọn các giá trị bị thổi phồng do sai số, "
         "khiến giá trị Q bị ước lượng quá cao một cách có hệ thống, làm chệch hướng chính sách. Double DQN giải quyết triệt để bằng cách tách rời: "
         "Mạng Online (theta) dùng để CHỌN hành động tốt nhất a* = argmax Q(s', a; theta), còn Mạng Target (theta^-) dùng để ĐÁNH GIÁ giá trị của a*: "
         "Y = R + gamma * Q(s', a*; theta^-)."),
        
        ("Câu 3: Kiến trúc Dueling DQN có ưu thế gì vượt trội so với DQN truyền thống trong game Mario?",
         "Dueling DQN phân rã Q(s, a) thành hai luồng độc lập: Giá trị trạng thái V(s) (trạng thái này tốt hay xấu) và Ưu thế hành động A(s, a) "
         "(hành động này vượt trội hơn hành động trung bình bao nhiêu). Trong Super Mario Bros, có rất nhiều trạng thái mà mọi hành động đều mang lại "
         "kết quả như nhau (ví dụ: Mario đang rơi tự do trên không trung hoặc đứng ở vùng đất hoàn toàn bằng phẳng không có quái vật, việc bấm nút hay không "
         "bấm nút không làm thay đổi giá trị của trạng thái). Dueling DQN cho phép mạng học giá trị V(s) trực tiếp và chính xác mà không cần phải trải nghiệm "
         "toàn bộ các hành động tại trạng thái đó, giúp tốc độ hội tụ nhanh hơn gấp nhiều lần."),
        
        ("Câu 4: Giải thích hiện tượng Mario bị 'kẹt ở ống nước' hoặc 'sợ chết nhảy lùi' và nhóm đã khắc phục bằng cách nào?",
         "Đây là hiện tượng rơi vào cực tiểu cục bộ (Local Minima). Khi gặp ống nước cao, Mario cần phải bấm tổ hợp phím chạy nhanh lấy đà và nhấn giữ phím nhảy. "
         "Khi chưa học được kỹ năng này, việc đâm vào ống nước hoặc đi tiếp sẽ đối mặt nguy cơ bị quái vật va chạm làm mất mạng (bị phạt nặng -50). "
         "Tác nhân phát hiện ra rằng việc đứng yên một chỗ hoặc chạy lùi lại sẽ an toàn hơn và không bị phạt chết. Nhóm đã khắc phục bằng 3 kỹ thuật: "
         "(1) Bổ sung Anti-stuck Penalty (phạt nếu không tăng tọa độ X quá 2 pixel trong 50 bước); (2) Phạt thời gian trên từng bước (-0.1); "
         "(3) Thưởng cực lớn khi chạm cờ (+500), định hướng tác nhân phải mạo hiểm vượt chướng ngại vật để tối đa hóa phần thưởng."),
        
        ("Câu 5: Tiêu chí nào chứng minh một cách khoa học rằng tác nhân thực sự giải được màn 1-1 chứ không phải ăn may?",
         "Nhóm chứng minh thông qua 3 bằng chứng định lượng khách quan: (1) Cờ chiến thắng flag_get == True được engine game NES xác nhận chính thức, "
         "tương đương tọa độ X >= 3161; (2) Tắt hoàn toàn khám phá ngẫu nhiên (epsilon = 0.0), chạy đánh giá trên 20 seeds ngẫu nhiên độc lập cố định "
         "bằng script evaluate.py, tỷ lệ thắng (Win Rate) đạt >= 80%; (3) Số bước hoàn thành màn chơi ổn định trong khoảng 250-300 steps (tương đương 18-20 giây "
         "thời gian thực của người chơi giỏi)."),
        
        ("Câu 6: Tại sao trong quá trình huấn luyện, giá trị Loss giảm rất thấp nhưng tác nhân vẫn không hoàn thành được màn chơi?",
         "Trong Deep Q-Learning, Loss chỉ đo lường mức độ khớp giữa dự đoán Q-online và mục tiêu Bellman Target (Y). Một giá trị Loss nhỏ chỉ chứng minh rằng "
         "mạng nơ-ron đang dự đoán rất chính xác giá trị của các cặp (s, a) mà nó ĐÃ TỪNG TRẢI NGHIỆM trong Replay Buffer. Nếu trong Replay Buffer chỉ toàn "
         "các mẫu trải nghiệm Mario bị chết ở đầu màn (X < 400), mạng nơ-ron sẽ dự đoán rất chính xác rằng 'mọi hành động ở đây đều có giá trị Q âm', "
         "khi đó Loss rất bé nhưng tác nhân hoàn toàn chưa hề khám phá được vùng đất phía sau ống nước. Do đó, chỉ số đánh giá thành công thực sự của bài toán RL "
         "phải là Episode Reward, Max X-Position và Win Rate, chứ không thể chỉ nhìn vào Loss.")
    ]

    for q, a in qa_list:
        add_h2(q)
        add_p(a, bold_prefix="Đáp án bản chất học thuật: ")
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # =========================================================================
    # PHẦN 14: TIÊU CHÍ NGHIỆM THU
    # =========================================================================
    add_h1("14. TIÊU CHÍ KIỂM TRA VÀ NGHIỆM THU SẢN PHẨM (ACCEPTANCE CRITERIA)")
    
    acc_table = doc.add_table(rows=11, cols=4)
    acc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    acc_table.autofit = False
    acc_table.columns[0].width = Cm(4.0)
    acc_table.columns[1].width = Cm(6.5)
    acc_table.columns[2].width = Cm(3.5)
    acc_table.columns[3].width = Cm(2.0)
    set_table_borders(acc_table)

    acc_headers = ["Hạng mục kiểm tra", "Chỉ số kỹ thuật bắt buộc", "Phương pháp kiểm tra", "Trách nhiệm"]
    for i, h in enumerate(acc_headers):
        c = acc_table.cell(0, i)
        set_cell_background(c, "003366")
        set_cell_margins(c, top=100, bottom=100, left=80, right=80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    acc_data = [
        ("Môi trường Gymnasium", "Chạy 1,000 steps liên tục không crash, RAM < 1.5GB", "Chạy tests/test_env.py", "Người 1"),
        ("Kiến trúc Mạng CNN", "Input (Batch, 4, 84, 84), Output (Batch, 7)", "Chạy Pytest test_model.py", "Người 2"),
        ("Tốc độ Huấn luyện", "Tốc độ >= 80 FPS trên GPU (hoặc >= 30 FPS trên CPU)", "Đo FPS thực tế trong train.py", "Người 2, 3"),
        ("Lưu & Tải Checkpoint", "Nạp checkpoint .pth không sai lệch tham số", "Chạy script nạp thử nghiệm", "Người 2"),
        ("MỤC TIÊU 8 ĐIỂM", "Mario chạm cờ flag_get == True, Win Rate >= 80% (20 seeds)", "Chạy evaluate.py", "Cả nhóm"),
        ("CẢI TIẾN 2 ĐIỂM", "Có số liệu đối sánh định lượng rõ ràng (Ablation/PPO)", "Kiểm tra log TensorBoard", "Người 3, 2"),
        ("Tính tái lập", "Cài đặt và chạy thành công trên máy tính thứ hai độc lập", "Chạy theo hướng dẫn README", "Người 4"),
        ("Video Demo HUD", "Định dạng MP4 full-HD, hiển thị Action, Q-value, Pos X", "Mở file trong recordings/", "Người 4"),
        ("Báo cáo học thuật", "12 chương, chuẩn hình thức Times New Roman, trích dẫn IEEE", "Rà soát file PDF hoàn chỉnh", "Người 4 (Lead)"),
        ("Diễn tập Thuyết trình", "4 thành viên trình bày khớp 20 phút, trả lời đúng 100% câu hỏi", "Thực hiện 2 buổi Mock Defense", "Cả nhóm")
    ]

    for row_idx, (cat, spec, meth, resp) in enumerate(acc_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate([cat, spec, meth, resp]):
            c = acc_table.cell(row_idx, col_idx)
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if col_idx in [0, 2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9)
            if col_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # PHẦN 15: QUẢN LÝ RỦI RO
    # =========================================================================
    add_h1("15. BẢNG QUẢN LÝ RỦI RO VÀ PHƯƠNG ÁN DỰ PHÒNG (CONTINGENCY PLANS)")
    
    risk_table = doc.add_table(rows=9, cols=5)
    risk_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    risk_table.autofit = False
    risk_table.columns[0].width = Cm(1.2)
    risk_table.columns[1].width = Cm(3.8)
    risk_table.columns[2].width = Cm(2.2)
    risk_table.columns[3].width = Cm(4.4)
    risk_table.columns[4].width = Cm(4.4)
    set_table_borders(risk_table)

    risk_headers = ["Mã", "Rủi ro tiềm ẩn", "Khả năng / Tác động", "Biện pháp phòng ngừa", "Phương án xử lý dự phòng (Contingency)"]
    for i, h in enumerate(risk_headers):
        c = risk_table.cell(0, i)
        set_cell_background(c, "003366")
        set_cell_margins(c, top=100, bottom=100, left=60, right=60)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    risk_data = [
        ("R01", "Lỗi cài đặt nes-py trên Python 3.12/3.13 (thiếu MSVC C++)", "Cao / Rất lớn", "Thống nhất dùng Python 3.10/3.11 trong môi trường Conda có sẵn wheel.", "Người 1 cung cấp file Dockerfile hoặc pre-built wheel cục bộ cho cả nhóm."),
        ("R02", "Xung đột API Gymnasium cũ/mới (step trả 4 hay 5 giá trị)", "Cao / Lớn", "Viết sẵn GymnasiumCompatibilityWrapper ngay từ Ngày 3 Tuần 1.", "Sử dụng shimmy wrapper hoặc ép kiểu step() trả về 5 giá trị thống nhất."),
        ("R03", "Mario bị kẹt ở ống nước cao, dậm chân tại chỗ", "Cao / Rất lớn", "Theo dõi TensorBoard hàng ngày; bổ sung Reward v2 có Anti-stuck.", "Tăng cường phạt đứng yên lũy tiến, thưởng nhảy khi tới gần tọa độ ống nước."),
        ("R04", "Thời gian huấn luyện quá lâu, thiếu tài nguyên GPU cá nhân", "Trung bình / Lớn", "Tối ưu hóa FrameSkip=4, kích thước ảnh 84x84; Replay buffer uint8.", "Chuyển pipeline lên Google Colab Pro / Kaggle GPU (T4/P100) chạy nền song song."),
        ("R05", "Hỏng hóc máy tính, mất checkpoint do cúp điện", "Thấp / Rất lớn", "Script tự động lưu checkpoint mỗi 50k steps và lưu top 3 model tốt nhất.", "Người 3 tự động đẩy checkpoint tốt lên Google Drive sau mỗi đợt chạy."),
        ("R06", "Xung đột mã nguồn (Merge Conflict) làm hỏng nhánh main", "Trung bình / Vừa", "Nghiêm cấm push trực tiếp vào main; mỗi PR chỉ giải quyết 1 tính năng.", "Người 4 chủ trì phiên gỡ conflict thủ công qua Git GUI hoặc VS Code."),
        ("R07", "Thành viên bị ốm hoặc chậm tiến độ nghiêm trọng", "Trung bình / Vừa", "Bảng phân công luôn có Người phối hợp (Backup) nắm rõ nghiệp vụ.", "Kích hoạt hỗ trợ chéo: Người 4 và Người phối hợp tiếp quản nhiệm vụ đó ngay."),
        ("R08", "Không kịp thời gian cho phần mở rộng 2 điểm cộng", "Trung bình / Thấp", "Đạt trọn vẹn 8 điểm P0 mới phân bổ tài nguyên cho P2.", "Nếu chậm, chỉ thực hiện 1 nghiên cứu triệt tiêu đơn giản nhất (Ablation Reward).")
    ]

    for row_idx, (code, desc, imp, prev, cont) in enumerate(risk_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate([code, desc, imp, prev, cont]):
            c = risk_table.cell(row_idx, col_idx)
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if col_idx in [0, 2]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            if col_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # PHẦN 16: CHECKLIST HOÀN THÀNH DỰ ÁN
    # =========================================================================
    add_h1("16. CHECKLIST TỔNG KẾT HOÀN THÀNH DỰ ÁN (FINAL SIGN-OFF)")
    add_p(
        "Trước khi chính thức nộp sản phẩm cho Giảng viên hướng dẫn, Trưởng nhóm và cả 4 thành viên phải cùng rà soát "
        "và ký xác nhận hoàn thành vào từng mục dưới đây:",
        bold_prefix="Quy trình nghiệm thu: "
    )
    add_bullet("[ ] Mã nguồn trên GitHub sạch sẽ, không chứa file rác cache __pycache__, có file requirements.txt chuẩn.", bold_prefix="1. Mã nguồn: ")
    add_bullet("[ ] Script evaluate.py chạy độc lập và tái lập chính xác kết quả Win Rate >= 80% trên 20 test episodes.", bold_prefix="2. Kết quả định lượng: ")
    add_bullet("[ ] Checkpoint best_model.pth được lưu trữ an toàn kèm link dự phòng Google Drive trong README.", bold_prefix="3. Trọng số mô hình: ")
    add_bullet("[ ] Script demo.py chạy mượt mà, hiển thị thanh thông số HUD trực quan và có sẵn video MP4 trong recordings/.", bold_prefix="4. Trực quan hóa: ")
    add_bullet("[ ] Báo cáo học thuật 12 chương đạt chuẩn hình thức A4, font Times New Roman, trích dẫn chuẩn IEEE.", bold_prefix="5. Báo cáo học thuật: ")
    add_bullet("[ ] Slide thuyết trình 25-28 trang hoàn thiện, thẩm mỹ cao, phân chia lời thoại cân bằng cho 4 người.", bold_prefix="6. Slide thuyết trình: ")
    add_bullet("[ ] Cả 4 thành viên đã thuộc lòng câu trả lời cho Ngân hàng câu hỏi vấn đáp và hoàn thành 2 buổi Mock Defense.", bold_prefix="7. Sẵn sàng vấn đáp: ")

    # Lưu tài liệu
    doc.save(output_path)
    print(f"[+] Tài liệu đã được tạo thành công tại: {output_path}")

if __name__ == "__main__":
    out_file = os.path.join(os.getcwd(), "Phan_cong_cong_viec_nhom_Mario_Gymnasium_5_tuan.docx")
    create_full_document(out_file)
