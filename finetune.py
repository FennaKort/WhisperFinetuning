import json
import os

from datasets import Audio, Dataset, DatasetDict
from transformers import WhisperProcessor, WhisperTokenizer
from whisper import load_audio, pad_or_trim # TODO 2026/07/20 need to integrate loading and storage of audio file to array and padding of audio array into data preparation for both split audio and sub-split-threshold

custom_dataset = DatasetDict()

# load json file to dictionary
# from dictionary, load to dataset
# ok so then that means that i need to update the way the json file is working first?? idk why i think that. 

class FineTuner:
	def __init__(self):
		self.metadata:list
	
	def get_metadata(self) -> list:
		return self.metadata

	def set_metadata(self, metadata: list) -> None:
		self.metadata = metadata

	def load_metadata_from_json(self, file_path: str) -> None:
		with open(file_path, 'r') as json_file:
			self.set_metadata(json.load(json_file))
	
	def prepare_dataset(self, batch):
		# load and resample audio data from 48 to 16kHz
		audio = batch["audio"]

		processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")

		# compute log-Mel input features from input audio array 
		batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

		# encode target text to label ids 
		batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
		return batch
	
	def make_dataset(self, json_metadata_path:str) -> Dataset:
		dataset = Dataset.from_json(json_metadata_path)

		# needs absolute filepath, array representing audio, and sampling_rate

		audio_column:list = []
		i:int = 0

		while i<dataset.num_rows:
			file_path = dataset["file_name"][i]
			audio_column.append(self.make_additional_audio_metadata(file_path, 16000)) #TODO 2026/07/20 figure out where is best to store sample rate as var instead
			i+=1

		dataset = dataset.remove_columns(["file_name","speech_ends_at","model_name","manually_verified"])
		dataset = dataset.add_column("audio", audio_column)

		print(dataset)
		print(dataset["transcript"][0])

		return dataset #.map(self.prepare_dataset, num_proc=4)
	
	def make_additional_audio_metadata(self, file_name:str, sampling_rate:int) -> dict:
		# 'audio': {'path': 'x', 'array': y, 'sampling_rate': x}
		script_dir = os.path.dirname(os.path.abspath(__file__))
		return {'path': script_dir+file_name, 'array': self.audio_to_padded_tensor(file_name, sampling_rate), 'sampling_rate': sampling_rate}

	def audio_to_padded_tensor(self, file_name:str, sampling_rate: int):
		audio_array = load_audio(file_name, sampling_rate) # use Whisper's load_audio() to convert audio to a NumPy array, resampling to provided sample rate if necessary
		audio_tensor = pad_or_trim(audio_array) # pads or trims audio array to a tensor of N_SAMPLES as expected by Whisper's encoder
		return audio_tensor #.tolist() # TODO 2026/07/20 this numpy tensor needs to be converted to a json serializable object somehow in order for it to output correctly. maybe answer is to just do this work inside the finetuning module lol?

def main():	
	finetuner = FineTuner()
	dataset = finetuner.make_dataset("res/validated-audio/metadata-subset.json")
	


if __name__ == "__main__":
	main()