# Agent Lab

**Tên:** Nguyễn Huy Tú  
**MSV:** 2A202600170

## Tôi đã làm gì

- Xây dựng benchmark so sánh `react` và `reflexion` cho bài toán multi-hop QA kiểu HotpotQA.
- Chạy đánh giá theo luồng có actor, evaluator và báo cáo kết quả đầu ra.
- Xuất report ở cả dạng JSON và Markdown để tiện kiểm tra và nộp bài.

## Cải thiện thêm

Các extensions đã triển khai:

- `structured_evaluator`
- `reflection_memory`
- `benchmark_report_json`
- `adaptive_max_attempts`

Tác động chính:

- Đánh giá nhất quán hơn nhờ output có cấu trúc.
- Agent nhớ lại lỗi trước đó để sửa trong các lần thử sau.
- Có report JSON phục vụ phân tích tự động.
- Dừng sớm khi lỗi lặp lại hoặc không còn khả năng cải thiện.

## Tài liệu liên quan

- [Sample report JSON](outputs/sample_run/report.json)
- [Sample report Markdown](outputs/sample_run/report.md)
- [Agent architecture](src/reflexion_lab/AGENTS_ARCHITECTURE.md)
