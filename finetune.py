import json

from datasets import Audio, Dataset, DatasetDict
from transformers import WhisperProcessor, WhisperTokenizer

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
		batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
		return batch
	
	def make_dataset(self, json_metadata_path:str) -> Dataset:
		dataset = Dataset.from_json(json_metadata_path).cast_column("file_name", Audio(sampling_rate=16000))
		dataset = dataset.remove_columns(["speech_ends_at","model_name","manually_verified"])
		dataset = dataset.rename_column("file_name","audio")
		print(dataset)
		print(dataset["transcript"][0])

		return dataset.map(self.prepare_dataset, num_proc=4)

def main():	
	finetuner = FineTuner()
	dataset = finetuner.make_dataset("res/validated-audio/metadata-subset.json")
	


if __name__ == "__main__":
	main()