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


def main():	
	dataset = Dataset.from_json("res/validated-audio/metadata.json").cast_column("file_name", Audio(sampling_rate=16000))
	dataset = dataset.remove_columns(["speech_ends_at","model_name","manually_verified"])
	dataset = dataset.rename_column("file_name","audio")
	print(dataset)

	# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
	

	processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
	
	print(dataset["transcript"][0])


if __name__ == "__main__":
	main()