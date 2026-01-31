import json, requests

url = "http://127.0.0.1:8000/analyze"
audio_path = r"./experiments/wav_data/i_like_to_dance_test.wav"

analysis = {
    "fullText": "I like to dance",
    "wordDetails": [
        {"text":"I","phonemes":[{"cpl":"AY","cipa":"aɪ","type":"vowel"}],"kor":{"ckor":"아"}},
        {"text":"like","phonemes":[{"cpl":"L","cipa":"l","type":"liquid"},{"cpl":"AY","cipa":"aɪ","type":"vowel"},{"cpl":"K","cipa":"k","type":"stop"}],"kor":{"ckor":"라이크"}},
        {"text":"to","phonemes":[{"cpl":"T","cipa":"t","type":"stop"},{"cpl":"UW","cipa":"u","type":"vowel"}],"kor":{"ckor":"투"}},
        {"text":"dance","phonemes":[{"cpl":"D","cipa":"d","type":"stop"},{"cpl":"AE","cipa":"æ","type":"vowel"},{"cpl":"N","cipa":"n","type":"nasal"},{"cpl":"S","cipa":"s","type":"fricative"}],"kor":{"ckor":"댄스"}}
    ]
}

with open(audio_path, "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    data = {"taskId": "req_test_001", "analysisRequest": json.dumps(analysis)}
    r = requests.post(url, files=files, data=data, stream=True)

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        obj = json.loads(line)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
