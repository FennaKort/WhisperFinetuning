import os
import numpy
from whisper import load_audio, pad_or_trim, load_model, transcribe
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from evaluate import load
from datasets import Audio, Dataset

from finetune import FineTuner, extract_features_and_tokenize

def set_dll_search_dir():
	on_path = os.environ.get('PATH')
	if on_path != None:
		on_path = on_path.split(";")
		for path in on_path:
			if ("FFmpeg" in path) and ("Shared" in path):
				os.add_dll_directory(path)
				print(f"{path} added as dll directory")

set_dll_search_dir()
import torch

def demo_load_audio_output():
	"""
	Demonstration of Whisper's load_audio() function showing how the size of the audio amplitude arrays returned by the function are affected by passing in different audio sampling rates. 
	
	Shows that the amplitude arrays are shorter when the function resamples audio at Whisper's native sampling rate of 16kHz, and that Whisper will automatically resample to this rate if no sampling rate is passed into load_audio(). Using this process therefore replicates the functionality of `datasets` `cast_column("audio", Audio(sampling_rate=16000))` method, which is the resampling approach recommended in most Whisper fine-tuning guides that use premade audio datasets that already have amplitude array data constructed from their audio files' native sampling rates. 
	"""
	audio_array = load_audio("res/audio/voice-message-1.mp3", sr=44100) #44.1kHz is a typical native sampling rate for audio, so I am using sr (samplingrate) = 44100 for demonstration purposes
	print(f"{len(audio_array)} amplitude samples when sampled at 44.1kHz") 
	# x samples, 1632384 samples when using demo audio "res/audio/voice-message-1.mp3"

	audio_array = load_audio("res/audio/voice-message-1.mp3", sr=16000) # Whisper internally operates at 16kHz
	print(f"{len(audio_array)} amplitude samples when sampled at 16kHz") 
	#y samples, 592248 samples when using demo audio "res/audio/voice-message-1.mp3"
	# y samples = x samples/2.75625 (because 44100/16000=2.75625)

	audio_array = load_audio("res/audio/voice-message-1.mp3") # if no sampling rate is provided, Whisper will use its default sampling rate of 16000/16kHz.
	print(f"{len(audio_array)} amplitude samples when sampled at default Whisper sample rate") 
	# z samples, same number of samples as y
	#592248 samples

def demo_pad_array():
	audio_array = load_audio("res/audio/voice-message-1.mp3") # if no sampling rate is provided, Whisper will load audio using its default sampling rate of 16000/16kHz.
	print(f"{len(audio_array)} amplitude samples when sampled at default Whisper sample rate") 
	#592248 samples

	audio_array = pad_or_trim(audio_array)
	print(f"{len(audio_array)} samples after trimming") # audio is longer than 30s/480000 samples, will be trimmed to 480000 samples, 16kHz sample rate * 30 second chunk size

	audio_array = load_audio("res/audio/voice-message-4.ogg") # if no sampling rate is provided, Whisper will use its default sampling rate of 16000/16kHz.
	print(f"{len(audio_array)} amplitude samples when sampled at default Whisper sample rate") 
	#158400 samples

	audio_array = pad_or_trim(audio_array)
	print(f"{len(audio_array)} samples after padding") # audio is shorter than 30s/480000 samples, will be padded or trimmed to 480000 samples, 16kHz sample rate * 30 second chunk size

def demo_compute_metrics():
	"Demos the word error rate (WER) calculation provided by Evaluate using the manually verified transcript for voice-message-1-split-1.mp3 as reference, compared to the predicted text output by default Whisper tiny.en model. For more information on WER in speech recognition, please see https://huggingface.co/learn/audio-course/en/chapter5/evaluation"
	metric = load("wer")
	reference = " Yeah, I totally get that. I absolutely love the energy of the storms. Like, yeah, of course, you know, there's damage or danger, like that's no good. But the energy? It's just such cool shit. We get these just like wild thunderstorms here and so often they'll just like rip through in like 15 to 30 minutes." # 57 words
	prediction = " Yeah, I totally get that. I absolutely love the energy. The storm is like, yeah, of course, you know, there's damage or danger, like that's no good. But with the energy, it's just such cool shit. We get these just like wild thunderstorms here. And so often they'll just like rips through in like 15 to 30 minutes." 
		# substitutions: 8 ('here. And' counted as one substitution)
		# insertions: 2
		# deletions: 1
		# expected WER: (8+2+1)/57~=0.192982

	wer = metric.compute(references=[reference], predictions=[prediction])

	print(wer) # 0.19298245614035087

	if wer != None:
		wer = wer * 100 # type: ignore
		
	print(wer)

	return {'wer': wer}


def demo_transcribe(input_features):
	model = load_model("tiny.en")
	result:dict = model.transcribe(input_features)
	print(result["text"])

def demo_compare_audio_data_creation_methods():
	conditional_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
	processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")

	def demo_cast_to_audio():
		print("Using Dataset's cast_column() to Audio feature type functionality:")
		dataset = Dataset.from_json("res/validated-audio/metadata-subset.json")
		dataset = dataset.rename_column("file_name","audio").cast_column("audio", Audio(sampling_rate=16000))

		dataset_mapped = dataset.map(extract_features_and_tokenize, remove_columns=["speech_ends_at","model_name","manually_verified"])

		input_features = dataset_mapped[0]["input_features"]
		print(input_features[0][0:10]) #-0.5177634954452515

		inputs = processor.feature_extractor(input_features, sampling_rate=16000, return_tensors="pt")
		input_features = inputs.input_features
		generated_ids = conditional_model.generate(input_features=input_features)
		transcript = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		print(transcript)

		
		return dataset

	def demo_make_dataset():
		print("\n Using finetune.py's make_dataset() method leveraging Whisper's native load and pad audio functions:")
		finetuner = FineTuner()
		dataset = finetuner.make_dataset("res/validated-audio/metadata-subset.json")
		input_features = dataset[0]["input_features"]
		print(input_features[0][0:10]) #-0.5177633762359619 
		return dataset

	def demo_batch_decode(labels, processor):
		transcript = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
		print(transcript)

	casted = demo_cast_to_audio()
	made = demo_make_dataset()
	print(casted["input_features"] == made["input_features"])
	print(casted["labels"] == made["labels"]) # ofc these are the same, they are the tokenized manually verified transcripts. 
	print(casted["input_features"][0][0][0])
	print(made["input_features"][0][0][0])

	print("casted transcripts:")
	print("from arrays:")
	# for feature in casted["input_features"]: #got float instead of ndarray
	# 	feature = numpy.array(feature)
	# 	demo_transcribe(feature)

	input_features = casted.input_features

	generated_ids = conditional_model.generate(inputs=input_features)
	demo_batch_decode(generated_ids)

	print("from verified labels:")
	for labels in casted["labels"]:
		demo_batch_decode(labels)

	print("\nmade transcripts:")
	print("from arrays:")
	demo_transcribe(numpy.array(made[0]['input_features']).astype('d'))
	# for feature in made["input_features"]: #got list instead of ndarray
	# 	feature = numpy.array(object=feature, dtype=numpy.double)
	# 	demo_transcribe(feature)
	print("from verified labels:")
	for labels in made["labels"]:
		demo_batch_decode(labels)




def main():
	# demo_load_audio_output()
	# demo_pad_array()
	# demo_compute_metrics()
	demo_compare_audio_data_creation_methods()


if __name__ == '__main__':
	main()