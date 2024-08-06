import funasr
# from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = funasr.AutoModel(
    model= "iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
)

res = model.generate(
    input="d:/vad_example.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    # merge_vad=True,  #
    # merge_length_s=15,
)

print("===============")
print(res)