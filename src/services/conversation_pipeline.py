import json
from src.models.conversation import handle_conversation_request

def conversation_stream(payload: dict):
    task_id = payload.get("taskId", "")
    try:
        result = handle_conversation_request(payload)

        # 이미 표준 포맷이면 그대로
        if all(k in result for k in ("type", "taskId", "status", "error", "analysisResult")):
            yield json.dumps(result, ensure_ascii=False) + "\n"
            return

        # 아니면 래핑
        yield json.dumps({
            "type": "conversation",
            "taskId": task_id,
            "status": "SUCCESS",
            "error": None,
            "analysisResult": result
        }, ensure_ascii=False) + "\n"

    except Exception as e:
        yield json.dumps({
            "type": "conversation",
            "taskId": task_id,
            "status": "FAIL",
            "error": str(e),
            "analysisResult": None
        }, ensure_ascii=False) + "\n"
