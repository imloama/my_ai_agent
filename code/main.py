# AI小助手完整代码
# 启动后，打开浏览器，进行对话
# 唤醒使用sherpa-onnx
# ASR使用funasr或fastasr
# llm使用ollama集成，qwen2 7B
# tts使用fish speech
import threading
import asyncio
import webrtcvad
import pyaudio
import funasr
import soundfile
import time
import sherpa_onnx
import webrtcvad
import numpy as np

# 音频数据队列
audio_queue_asr = asyncio.Queue() # ASR识别队列
audio_queue_kws = asyncio.Queue() # 唤醒识别队列
# 监听状态 0表示未唤醒 1表示已唤醒， -1表示停止
status = 0
keyword_spotter:sherpa_onnx.KeywordSpotter = None


vad:webrtcvad.Vad = webrtcvad.Vad(3)

# 监听麦克风，将数据保存到audio_queue
def listen_microphone():
    global audio_queue_asr
    global status
    global vad
    audio = pyaudio.PyAudio()
    chunk = 320 #4800 #3200 #1024  int(RATE * 0.2)  # 每次读取200毫秒的数据
    format = pyaudio.paInt16
    stream = audio.open(format=format,channels=1, rate=16000,input=True, frames_per_buffer=chunk)
    frames = b''
    while True:
        if status < 0:
            break
        data = stream.read(chunk)
        if status==0:
            audio_queue_kws.put_nowait(data)
            continue
        # TODO 判断是否为人声，不是人声直接抛弃 
        if vad.is_speech(data, 16000):
            frames += data
            continue
        if len(frames) > 0:
            audio_queue_asr.put_nowait(frames)
            frames = b''

# 初始化sherpa-onnx kws
def kws_init():
    global keyword_spotter
    model_path = "../models/sherpa-onnx-kws"
    keyword_spotter = sherpa_onnx.KeywordSpotter(
        tokens=f'{model_path}/tokens.txt',
        encoder=f'{model_path}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx',
        decoder=f'{model_path}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
        joiner=f'{model_path}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx',
        num_threads=2,
        max_active_paths=4,
        keywords_file=f'{model_path}/keywords.txt',
        keywords_score=3.5,
        keywords_threshold=0.2,
        provider='cpu',
    )
    # stream = keyword_spotter.create_stream()
async def do_kws():
    global keyword_spotter
    stream = keyword_spotter.create_stream()
    while True:
        if status < 0:
            break
        if audio_queue_kws.empty():
            await asyncio.sleep(0.2)
            continue
        data = await audio_queue_asr.get()
        frames_np = np.frombuffer(data, dtype=np.int16)
        # 进行采样率和数据类型的调整
        speech_chunk = frames_np.astype(np.float32)  # 调整数据类型
        # print(speech_chunk)
        #print("数组形状：",speech_chunk.shape)
        stream.accept_waveform(16000, speech_chunk)
        while keyword_spotter.is_ready(stream):
            keyword_spotter.decode_stream(stream)
        result = keyword_spotter.get_result(stream)
        if result == None or result == '':
            return
        # TODO 唤醒结果

# 从队列获取音频数据，执行ASR或唤醒处理
async def do_asr():
    global audio_queue_asr
    global status
    #webrtc_vad_model = webrtcvad.Vad()
    #webrtc_vad_model.set_mode(3) # 0: 高度不敏感（最少误报，但可能漏掉一些语音）1: 不太敏感2: 中等敏感  3: 高度敏感（最多误报，但最少漏掉语音）
    vad_model = funasr.AutoModel(model="../models/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    frames = b''
    cache = {}
    is_final = False
    while True:
        if status != 1:
            break
        print("==============1============")
        if audio_queue_asr.empty():
            await asyncio.sleep(0.2)
            continue
        data = await audio_queue_asr.get()
        # frames = frames + data
        # vad判断是否结束
        #webrtc_vad_model.is_speech()
        res = vad_model.generate(input=data, cache=cache, is_final=is_final, chunk_size=320)
        # if len(res[0]["value"]):
            # print(res)
        values = res[0]["value"]
        if len(values)  == 0:
            continue
        value = values[len(values)-1]
        end_value = value[1]
        if end_value < 0:
            # 还有数据
            frames = frames + data
        else:
            #todo 判断是否进行唤醒，还是进行语音识别
            # https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/vad-asr-paraformer/main.go
            # kws https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
            # kws https://github.com/k2-fsa/icefall/pull/1428
            
            frames = b''
            cache = {}
            
        print(res)
        # [{'key': 'rand_key_BkaYjKcCiv7a8', 'value': [[-1, 46900]]}]
        # [{'key': 'rand_key_UE68Q0QXqVNme', 'value': [[47310, -1]]}]
        # [{'key': 'rand_key_DAlbDZqm8yewg', 'value': [[-1, 56460], [56740, -1]]}]
        # 执行ASR，再执行获取拼音，判断是否为唤醒词
        

def do_asr_sync():
    asyncio.run(do_asr())

if __name__ == '__main__':
    status =  1
    rec = threading.Thread(target=do_asr_sync)
    rec.daemon = True
    rec.start()
    
    # listen_microphone()
    
    chunk_size = 320 # ms
    wav_file = f"../models/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav"
    speech, sample_rate = soundfile.read(wav_file)
    chunk_stride = int(chunk_size * sample_rate / 1000)
    cache = {}
    total_chunk_num = int(len((speech)-1)/chunk_stride+1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
        is_final = i == total_chunk_num - 1
        # res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
        # if len(res[0]["value"]):
        #     print(res)
        audio_queue_asr.put_nowait(speech_chunk) 
    
    time.sleep(10)