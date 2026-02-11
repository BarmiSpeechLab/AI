import json, requests
import argparse

url = "http://127.0.0.1:8000/analyze"

CASES = {
    "sentence": {
        "audio_path": r"./experiments/wav_data/i_like_to_dance_test.wav",
        "analysis": {
            "type": "sentence",
            "fullText": "I like to dance",
            "wordDetails": [
                {"text":"I","phonemes":[{"cpl":"AY","cipa":"aɪ","type":"vowel"}],"kor":{"ckor":"아"}},
                {"text":"like","phonemes":[{"cpl":"L","cipa":"l","type":"liquid"},{"cpl":"AY","cipa":"aɪ","type":"vowel"},{"cpl":"K","cipa":"k","type":"stop"}],"kor":{"ckor":"라이크"}},
                {"text":"to","phonemes":[{"cpl":"T","cipa":"t","type":"stop"},{"cpl":"UW","cipa":"u","type":"vowel"}],"kor":{"ckor":"투"}},
                {"text":"dance","phonemes":[{"cpl":"D","cipa":"d","type":"stop"},{"cpl":"AE","cipa":"æ","type":"vowel"},{"cpl":"N","cipa":"n","type":"nasal"},{"cpl":"S","cipa":"s","type":"fricative"}],"kor":{"ckor":"댄스"}}
            ]
        }
    },
    "word": {
        "audio_path": r"./experiments/wav_data/apple.wav",
        "analysis": {
            "type": "word",
            "fullText": "apple",
            "wordDetails": [
                {
                    "text": "apple",
                    "phonemes": [
                        {"cpl": "AE", "cipa": "æ", "type": "vowel"},
                        {"cpl": "P",  "cipa": "p", "type": "stop"},
                        {"cpl": "AH", "cipa": "ə", "type": "vowel"},
                        {"cpl": "L",  "cipa": "l", "type": "liquid"}
                    ],
                    "kor": {"ckor": "애플"}
                }
            ]
        }
    }
}


parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["sentence", "word"], default="sentence")
args = parser.parse_args()

audio_path = CASES[args.case]["audio_path"]
analysis = CASES[args.case]["analysis"]


with open(audio_path, "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    data = {"taskId": "req_test_001", "analysisRequest": json.dumps(analysis)}
    r = requests.post(url, files=files, data=data, stream=True)

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        obj = json.loads(line)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
