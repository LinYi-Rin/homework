import whisper
import os

def speech_to_text(audio_path):
    """
    使用 Whisper 进行语音识别
    :param audio_path: 音频路径
    :return: 识别文本
    """
    # 加载小模型（速度快，适合PC）
    model = whisper.load_model("base")

    # 识别音频
    result = model.transcribe(audio_path, language="zh")

    # 输出结果
    print("\n===== 语音识别结果 =====")
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    # 请把你的剪映音频放在同一目录下
    AUDIO_FILE = "output.wav"  # 你的配音文件

    if not os.path.exists(AUDIO_FILE):
        print(f"错误：未找到音频文件 {AUDIO_FILE}")
    else:
        speech_to_text(AUDIO_FILE)
