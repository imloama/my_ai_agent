# funasr的使用示例
# 说明：
#    关于模型的下载：国内网络，modelscope下载相对比较快，可以通过程序自动下载，下载目录位于`MODELSCOPE_CACHE`目录

# 1. 环境安装 

# pip install -r requirements.txt
# 到https://pytorch.org/get-started/locally/，确认安装哪个版本的torch和torchaudio

# 2. 代码执行

import time
from funasr import AutoModel

start_time = time.time()
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
# paraformer-zh是ASR模型，fsmn-vad是vad（语音端口检测），punc model是标点符号恢复
model = AutoModel(model="paraformer-zh",  #vad_model="fsmn-vad", punc_model="ct-punc", 
                  # spk_model="cam++"
                  )
# 参数，hotword表示热词，有些相近的语音会识别成热词，多个热词用空格分隔
res = model.generate(input="../../data/asr_example_cn_en.wav", 
            batch_size_s=300, 
            hotword='machine learning deep learning')
end_time = time.time()
print(res)
print("Model inference takes {:.2}s.".format(end_time - start_time))