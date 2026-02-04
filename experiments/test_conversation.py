import json, requests, argparse

url = "http://127.0.0.1:8000/conversation"

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="I_am_a_student.wav")  # uploads 기준
parser.add_argument("--taskId", default="CONV_test_001")
parser.add_argument("--theme", default="DAILY")
parser.add_argument("--prev", default="What do you do?")
parser.add_argument("--status", default="SUCCESS")
args = parser.parse_args()

payload = {
    "taskId": args.taskId,
    "filePath": args.file,
    "status": args.status,
    "error": None,
    "analysisRequest": {
        "prevTurn": args.prev,
        "theme": args.theme
    }
}

r = requests.post(url, json=payload, stream=True)

for line in r.iter_lines(decode_unicode=True):
    if not line:
        continue
    obj = json.loads(line)
    print(json.dumps(obj, ensure_ascii=False, indent=2))
