import whisper

# print(whisper.available_models()) 
# have verified what whisper models are available

model = whisper.load_model("base")
# result = model.transcribe("voice-message-3.mp3")
# ok so seems like the issue is the process not finding some subprocess executable in the python libraries? maybe this would be solvable by setting up a venv so everything is just right there???
# it's not about the file size/audio length because I switched to this version that's exactly 30s

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("./res/audio/voice-message-1.mp3") # ok so the issue is the FileNotFoundError is occurring here, when Whisper tries to load the audio file. Lumo says whisper calls ffmpeg via subprocess when running whisper.load_audio(), so this issue occurs when ffmpeg isn't installed or isn't in system PATH
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

