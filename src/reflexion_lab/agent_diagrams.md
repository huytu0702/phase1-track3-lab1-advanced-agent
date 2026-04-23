# Reflexion Lab - React & Reflexion Diagrams

Tài liệu này tóm tắt cách lab được cài đặt và cách hai agent hoạt động trong code hiện tại.

## Cài đặt nhanh

- Tạo môi trường ảo và cài dependency:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Với `real mode`, cần các biến môi trường:
  - `DEFAULT_MODEL`
  - `DEFAULT_BASE_URL`
  - `DEFAULT_API_KEY`
  - `JUDGE_MODEL`
  - tùy chọn: `JUDGE_BASE_URL`, `JUDGE_API_KEY`

- Chạy benchmark qua `run_benchmark.py`.
  - `mock` mode dùng runtime giả để smoke test.
  - `real` mode dùng OpenAI-compatible runtime.
  - Kết quả được ghi ra `react_runs.jsonl`, `reflexion_runs.jsonl`, `report.json`, `report.md`.

## Mô tả ngắn gọn kiến trúc

- `run_benchmark.py` nạp dataset, khởi tạo runtime, rồi chạy song song `ReActAgent` và `ReflexionAgent` trên cùng bộ câu hỏi.
- `BaseAgent.run()` là nơi điều phối vòng lặp: actor -> evaluator -> dừng hoặc phản tư.
- `ReActAgent` chỉ cho phép 1 lượt trả lời.
- `ReflexionAgent` có nhiều lượt thử, giữ `reflection_memory`, và có thể dừng sớm nếu lỗi lặp lại hoặc không còn khả năng sửa.
- `OpenAICompatibleRuntime` gọi 3 vai trò:
  - `actor`: trả lời câu hỏi từ context
  - `evaluator`: chấm câu trả lời bằng JSON có cấu trúc
  - `reflector`: sinh bài học và chiến lược cho lượt sau

## Flow Diagram

```mermaid
flowchart TD
    A[load dataset] --> B[build runtime<br/>mock or openai-compatible]
    B --> C[create ReActAgent]
    B --> D[create ReflexionAgent]
    C --> E[run examples in batches]
    D --> E
    E --> F[save react_runs.jsonl]
    E --> G[save reflexion_runs.jsonl]
    F --> H[build report]
    G --> H
    H --> I[save report.json]
    H --> J[save report.md]

    subgraph React[ReActAgent]
        R1[actor attempt 1] --> R2[evaluator]
        R2 --> R3{score == 1?}
        R3 -- yes --> R4[stop and record success]
        R3 -- no --> R5[stop after single attempt]
    end

    subgraph Reflexion[ReflexionAgent]
        X1[actor attempt N] --> X2[evaluator]
        X2 --> X3{score == 1?}
        X3 -- yes --> X4[stop and record success]
        X3 -- no --> X5{adaptive stop?}
        X5 -- unsalvageable / repeated failure --> X6[stop early]
        X5 -- continue --> X7[reflector creates lesson]
        X7 --> X8[append reflection_memory]
        X8 --> X1
    end
```

## Sequence Diagram - ReAct

```mermaid
sequenceDiagram
    autonumber
    participant Runner as run_benchmark.py
    participant Agent as ReActAgent
    participant Runtime as OpenAICompatibleRuntime / MockRuntime

    Runner->>Agent: run(example)
    Agent->>Runtime: actor(example, attempt_id=1, reflection_memory=[], trajectory=[])
    Runtime-->>Agent: answer
    Agent->>Runtime: evaluator(example, answer, attempt_id=1, ...)
    Runtime-->>Agent: JudgeResult + call metrics
    alt score == 1
        Agent-->>Runner: RunRecord success
    else score == 0
        Agent-->>Runner: RunRecord failure after 1 attempt
    end
```

## Sequence Diagram - Reflexion

```mermaid
sequenceDiagram
    autonumber
    participant Runner as run_benchmark.py
    participant Agent as ReflexionAgent
    participant Runtime as OpenAICompatibleRuntime / MockRuntime

    Runner->>Agent: run(example)
    loop until success or max_attempts
        Agent->>Runtime: actor(example, attempt_id, reflection_memory, trajectory)
        Runtime-->>Agent: answer
        Agent->>Runtime: evaluator(example, answer, attempt_id, ...)
        Runtime-->>Agent: JudgeResult + call metrics
        alt score == 1
            Agent-->>Runner: RunRecord success
        else score == 0
            alt adaptive stop triggers
                Note over Agent,Runner: stop early
            else continue
                Agent->>Runtime: reflector(example, answer, judge, attempt_id, reflection_memory, trajectory)
                Runtime-->>Agent: ReflectionEntry + call metrics
                Note over Agent: append reflection_memory and trajectory
            end
        end
    end
```

## Cách hoạt động thực tế

1. Dataset được nạp từ JSON HotpotQA-style và cắt theo batch.
2. Mỗi batch được chạy qua cả ReAct và Reflexion trên cùng runtime.
3. `evaluator` trả về JSON có cấu trúc để chuẩn hóa chấm điểm.
4. Với Reflexion, phần `reflection_memory` được truyền sang lượt sau để sửa lỗi suy luận nhiều hop hoặc entity drift.
5. Sau khi chạy xong, hệ thống tổng hợp EM, số lượt thử, token, latency, cost, rồi lưu báo cáo Markdown/JSON.
