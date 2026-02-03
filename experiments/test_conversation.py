import json, requests, argparse

url = "http://127.0.0.1:8000/conversation"

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="I_am_a_dudun.wav")        # uploads 폴더 기준
parser.add_argument("--taskId", default="req_test_001")
args = parser.parse_args()

payload = {
    "taskId": args.taskId,
    "file": args.file
}

r = requests.post(url, json=payload, stream=True)

for line in r.iter_lines(decode_unicode=True):
    if not line:
        continue
    obj = json.loads(line)
    print(json.dumps(obj, ensure_ascii=False, indent=2))
