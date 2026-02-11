import json, requests, argparse

url = "http://127.0.0.1:8000/conversation"

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="./src/uploads/hello.wav")
parser.add_argument("--taskId", default="CONV_test_001")
parser.add_argument("--theme", default="DAILY")
parser.add_argument("--prev", default="What do you do?")
args = parser.parse_args()

analysis_request = {
    "prevTurn": args.prev,
    "theme": args.theme
}

with open(args.file, "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    data = {
        "taskId": args.taskId,
        "analysisRequest": json.dumps(analysis_request)
    }
    r = requests.post(url, files=files, data=data, stream=True)

for line in r.iter_lines(decode_unicode=True):
    if not line:
        continue
    obj = json.loads(line)
    print(json.dumps(obj, ensure_ascii=False, indent=2))
