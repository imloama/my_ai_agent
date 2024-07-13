import fastasr
import soundfile as sf
import time
start_time = time.time()
fastasr_model_path = "./models/paraformer"
# 第2个值，3表示paramformer
fastasr_model = fastasr.Model(fastasr_model_path, 3)

data, samplerate = sf.read("../../data/asr_example_cn_en.wav", dtype='int16')
fastasr_model.reset()
result =fastasr_model.forward(data)
end_time = time.time()
print('Result: "{}".'.format(result))
print("Model inference takes {:.2}s.".format(end_time - start_time))