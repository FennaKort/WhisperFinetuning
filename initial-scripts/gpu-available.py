import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
x = torch.rand(5, 3)
print(x)

import whisper
print('Whisper is ready!')

import ffmpeg # ffmpeg-python wrapper
print('FFmpeg is ready!')