# augmentor.py
import torch
import torchaudio
import torchaudio.transforms as T


# 添加白噪声
def add_white_noise(audio, noise_level=0.005):
    noise = noise_level * torch.randn_like(audio)
    return audio + noise


# 时间偏移
def time_shift(audio, shift_max=0.2):
    shift_amount = int(audio.shape[-1] * shift_max * (2 * torch.rand(1) - 1))  # 随机偏移量
    return torch.roll(audio, shifts=shift_amount, dims=-1)


# 频率遮盖（SpecAugment）
def frequency_masking(audio, sample_rate, freq_mask_param=15):
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate)(audio)
    freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    return freq_mask(spectrogram)


# 时间遮盖（SpecAugment）
def time_masking(audio, sample_rate, time_mask_param=35):
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate)(audio)
    time_mask = T.TimeMasking(time_mask_param=time_mask_param)
    return time_mask(spectrogram)


# 改变音调
def pitch_shift(audio, sample_rate, n_steps=2):
    return torchaudio.functional.pitch_shift(audio, sample_rate, n_steps=n_steps)


# 改变速度
def time_stretch(audio, rate=1.1):
    # rate > 1.0 会加速音频，rate < 1.0 会减速音频
    return T.TimeStretch()(audio, rate)


# 混合音频（Mixup）
def mixup(audio1, audio2, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    return lam * audio1 + (1 - lam) * audio2


# 综合音频增强器类
class AudioAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio):
        # 以一定概率应用不同的增强方法
        if torch.rand(1).item() > 0.5:
            audio = add_white_noise(audio)
        if torch.rand(1).item() > 0.5:
            audio = time_shift(audio)
        if torch.rand(1).item() > 0.5:
            audio = frequency_masking(audio, self.sample_rate)
        if torch.rand(1).item() > 0.5:
            audio = time_masking(audio, self.sample_rate)
        if torch.rand(1).item() > 0.5:
            audio = pitch_shift(audio, self.sample_rate)
        return audio


# 示例用法
if __name__ == "__main__":
    sample_rate = 16000  # 根据数据集的采样率
    audio, sr = torchaudio.load("path/to/audio.wav")  # 加载音频文件
    augmentor = AudioAugmentor(sample_rate)

    # 增强音频
    augmented_audio = augmentor(audio)
    torchaudio.save("path/to/augmented_audio.wav", augmented_audio, sample_rate)
