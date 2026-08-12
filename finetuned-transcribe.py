from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper import load_audio
import torch
import os

def transcribe_audio(audio_path, processor, model, device):
    # Load the audio file
    sr=16000
    audio = load_audio(audio_path, sr=sr)  

    # Prepare the audio input for the model
    input_features = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    language_token_id = processor.tokenizer.convert_tokens_to_ids('<|en|>')
    forced_decoder_ids = [[1, language_token_id]]

    # Generate transcription with forced_decoder_ids for English
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Decode the transcription
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define your audio from audio_path
audio_path = "res/post-training-test/post-training-test-01.mp3"

# transcription with HF whisper tiny.en
hf_dir = "openai/whisper-tiny.en"
model = WhisperForConditionalGeneration.from_pretrained(hf_dir)
model.to(device)
processor = WhisperProcessor.from_pretrained(hf_dir)

print("Transcribing with HF whisper tiny.en:")
transcription = transcribe_audio(audio_path, processor, model, device)
print(transcription)
print("\n")


# transcription with the finetuned model and processor
fine_tuned_dir = "./fine-tuned-model/"
model = WhisperForConditionalGeneration.from_pretrained(fine_tuned_dir)
model.to(device)
processor = WhisperProcessor.from_pretrained(fine_tuned_dir)

print("Transcribing with finetuned model:")
transcription = transcribe_audio(audio_path, processor, model, device)
print(transcription)
