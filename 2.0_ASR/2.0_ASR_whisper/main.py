# 1. 创建环境

# 在当前目录的命令行中执行：
# 1.1 conda create -n fasterwhisper python=3.10 -y
# 1.2 conda activate fasterwhisper

# 2. 安装依赖

# pip install -r requirements.txt

# 3. 下载模型，模型下载完成后，即可注释代码

# 根据系统及自身要求选择不同的版本，本次测试使用int8 small版本进行演示

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download
#hf_hub_download(repo_id="Systran/faster-whisper-small", filename="config.json", local_dir="./models")
#hf_hub_download(repo_id="Systran/faster-whisper-small", filename="model.bin", local_dir="./models")
#hf_hub_download(repo_id="Systran/faster-whisper-small", filename="tokenizer.json", local_dir="./models")
#hf_hub_download(repo_id="Systran/faster-whisper-small", filename="vocabulary.txt", local_dir="./models")


# 4. cpu推理/gpu推理
from faster_whisper import WhisperModel

def test_for_wav_file():
    model_path = "./models"
    model = WhisperModel(model_path, device="cpu", compute_type="int8")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("../../data/asr_example_cn_en.wav", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        
        

# 测试流式输出
def test_for_wav_stream():
    pass

# 测试cpu推理，推理单个WAV文件
test_for_wav_file()