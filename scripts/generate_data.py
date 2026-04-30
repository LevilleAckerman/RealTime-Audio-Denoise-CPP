# scripts/generate_data.py

import numpy as np
import wave
import struct
import os

def generate_noisy_audio():
    # 1. 参数设置
    sample_rate = 16000       # 采样率 16kHz (语音常用)
    duration = 5              # 时长 5秒
    freq_clean = 400          # 模拟人声频率 (Hz)
    noise_amplitude = 0.5     # 噪音强度

    # 2. 生成时间轴
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 3. 生成“干净”信号 (模拟人声: 简单的正弦波)
    clean_signal = np.sin(2 * np.pi * freq_clean * t)

    # 4. 生成“噪音” (高斯白噪声)
    noise = np.random.normal(0, noise_amplitude, clean_signal.shape)

    # 5. 混合信号 (带噪音频)
    noisy_signal = clean_signal + noise

    # 6. 归一化防止溢出
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

    # 7. 保存为 WAV 文件 (16-bit PCM)
    output_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'noisy_input.wav')
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with wave.open(output_filename, 'w') as wav_file:
        wav_file.setnchannels(1)          # 单声道
        wav_file.setsampwidth(2)          # 2字节 (16-bit)
        wav_file.setframerate(sample_rate)

        for sample in noisy_signal:
            # 将浮点数 (-1.0 ~ 1.0) 转换为 16-bit 整数
            wav_file.writeframes(struct.pack('<h', int(sample * 32767.0)))

    print(f"✅ 成功生成测试音频: {output_filename}")

if __name__ == "__main__":
    generate_noisy_audio()