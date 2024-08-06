# AI小助手完整代码
# 启动后，打开浏览器，进行对话
# 唤醒使用sherpa-onnx
# ASR使用funasr或fastasr
# llm使用ollama集成，qwen2 7B
# tts使用fish speech
import threading
import asyncio
import funasr.auto
import webrtcvad
import pyaudio
import funasr
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import soundfile
import time
import sherpa_onnx
import webrtcvad
import numpy as np
from ollama import AsyncClient


ollama_host = 'http://localhost:11434'
ollama_model='qwen2:1.5b'

# 音频数据队列
audio_queue_asr = asyncio.Queue() # ASR识别队列
audio_queue_kws = asyncio.Queue() # 唤醒识别队列
# 监听状态 0表示未唤醒 1表示已唤醒， -1表示停止
status = 0
keyword_spotter:sherpa_onnx.KeywordSpotter = None
is_listening = True
is_chatting = False
last_kws_active_at = -1 #上次唤醒时的时间

vad:webrtcvad.Vad = webrtcvad.Vad(3)
asr_model:funasr.AutoModel = None

# 监听麦克风，将数据保存到audio_queue
def listen_microphone():
    global audio_queue_asr
    global status
    global vad
    global is_listening
    audio = pyaudio.PyAudio()
    chunk = 320 #4800 #3200 #1024  int(RATE * 0.2)  # 每次读取200毫秒的数据
    format = pyaudio.paInt16
    stream = audio.open(format=format,channels=1, rate=16000,input=True, frames_per_buffer=chunk)
    frames = b''
    while True:
        if status < 0:
            break
        if not is_listening:
            frames = b''
            time.sleep(0.1)
            continue
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

async def on_kws_result(cmd=None):
    global status
    global is_listening
    global audio_queue_asr
    global last_kws_active_at
    last_kws_active_at = time.time()
    status = 1
    is_listening = False
    while not audio_queue_asr.empty():
        audio_queue_asr.get_nowait()
    # TODO 播放声音
    await asyncio.sleep(2)
    is_listening = True
    
async def on_asr_result(result):
    global status
    global is_listening
    global is_chatting
    global last_kws_active_at
    last_kws_active_at = time.time()
    is_listening = False
    # TODO 请求ollama，得到回答
    is_chatting = True
    resp = await chat_by_ollama(result)
    # tts
    
    #TODO 恢复
    is_listening = True

async def chat_by_ollama(question):
    message = {'role': 'user', 'content': question}
    response = await AsyncClient(host=ollama_host).chat(model=ollama_model, messages=[message])
    print(response)
    return response["message"]["content"]

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
        await on_kws_result(result)

# 从队列获取音频数据，执行ASR或唤醒处理
async def do_asr():
    global audio_queue_asr
    global status
    global asr_model
    asr_model = funasr.AutoModel(model= "iic/SenseVoiceSmall",
                                 trust_remote_code=True,
    remote_code="./model.py",
    #vad_model="fsmn-vad",
    #vad_kwargs={"max_single_segment_time": 30000},
    #device="cpu"#"cuda:0",
    )
    
    while True:
        if audio_queue_asr.empty():
            await asyncio.sleep(0.2)
            continue
        data = await audio_queue_asr.get()
        res = asr_model.generate(
            input=data,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        print(res)
        text = rich_transcription_postprocess(res[0]["text"])
        print(text)
        #todo 打印
        await on_asr_result(text)
        

def do_asr_sync():
    asyncio.run(do_asr())
    
def do_kws_sync():
    asyncio.run(do_kws())
    
def do_kws_reset():
    global last_kws_active_at
    global status
    while True:
        if status == 0:
            time.sleep(0.2)
            continue
        if time.time() - last_kws_active_at > 600:
            status = 0
            time.sleep(1)
            continue
        

if __name__ == '__main__':
    status =  1
    thread_asr = threading.Thread(target=do_asr_sync)
    thread_asr.daemon = True
    thread_asr.start()
    
    thread_kws = threading.Thread(target=do_kws_sync)
    thread_kws.daemon = True
    thread_kws.start()
    
    thread_reset_kws = threading.Thread(target=do_kws_reset)
    thread_reset_kws.daemon = True
    thread_reset_kws.start()
    
    
    
    listen_microphone()
    
    '''
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
    '''