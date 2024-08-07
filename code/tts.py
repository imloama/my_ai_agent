import edge_tts
import asyncio
from pydub import AudioSegment
from pydub.playback import play


#zh-CN-XiaoxiaoNeural	汉语（简体中文） 普通话	中国	女性
#zh-CN-XiaoyiNeural	汉语（简体中文） 普通话	中国	女性

edge_tts_voice = "zh-CN-XiaoxiaoNeural"

async def get_edge_tts_voices():
    voices = await  edge_tts.list_voices()
    print(voices)
    return voices


async def edge_tts_play(content, filename):
    communicate = edge_tts.Communicate(content, edge_tts_voice)
    # for chunk in communicate.stream_sync():
    #         if chunk["type"] == "audio":
    #             audio_data = AudioSegment(data=chunk["data"], sample_width=2, frame_rate=16000, channels=1)
    #             # 使用pydub播放音频
    #             play(audio_data)
    #             print("=-------------------")
    #         elif chunk["type"] == "WordBoundary":
    #             print(f"WordBoundary: {chunk}")
    await communicate.save(filename)



if __name__ == '__main__':
#    asyncio.run(get_edge_tts_voices())
    asyncio.run(edge_tts_play("我在，主人","./kws.mp3"))
    audio = AudioSegment.from_mp3("./kws.mp3")
    play(audio)