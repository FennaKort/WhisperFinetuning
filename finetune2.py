import pathlib

from datasets import Audio, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

dataset = Dataset.from_json("res/validated-audio/metadata-subset.json")
dataset = dataset.remove_columns(["speech_ends_at","model_name","manually_verified"])
dataset = dataset.rename_column("file_name","audio")

print(dataset)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")

input_str = dataset["transcript"][0]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

print(dataset["audio"][0])

dataset.cast_column("audio", Audio(sampling_rate=16000))

dataset["audio"][0]
