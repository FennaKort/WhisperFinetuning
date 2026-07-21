import json
import os
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Audio, Dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from whisper import load_audio, pad_or_trim # TODO 2026/07/20 need to integrate loading and storage of audio file to array and padding of audio array into data preparation for both split audio and sub-split-threshold

custom_dataset = DatasetDict()

# load json file to dictionary
# from dictionary, load to dataset
# ok so then that means that i need to update the way the json file is working first?? idk why i think that. 

class FineTuner:
	def __init__(self):
		self.metadata:list
		self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe") # TODO 2026/07/20 would love to figure out loading from local model 

	
	def get_metadata(self) -> list:
		return self.metadata

	def set_metadata(self, metadata: list) -> None:
		self.metadata = metadata

	def load_metadata_from_json(self, file_path: str) -> None:
		with open(file_path, 'r') as json_file:
			self.set_metadata(json.load(json_file))
	
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

		return dataset.map(self.prepare_dataset, num_proc=4)
	
	def make_additional_audio_metadata(self, file_name:str, sampling_rate:int) -> dict:
		# 'audio': {'path': 'x', 'array': y, 'sampling_rate': x}
		script_dir = os.path.dirname(os.path.abspath(__file__))
		return {'path': script_dir+file_name, 'array': self.audio_to_padded_tensor(file_name, sampling_rate), 'sampling_rate': sampling_rate}

	def audio_to_padded_tensor(self, file_name:str, sampling_rate: int):
		audio_array = load_audio(file_name, sampling_rate) # use Whisper's load_audio() to convert audio to a NumPy array, resampling to provided sample rate if necessary
		audio_tensor = pad_or_trim(audio_array) # pads or trims audio array to a tensor of N_SAMPLES as expected by Whisper's encoder
		return audio_tensor # TODO 2026/07/20 this numpy tensor needs to be converted to a json serializable object somehow in order for it to output correctly. maybe answer is to just do this work inside the finetuning module lol?
		
	def prepare_dataset(self, batch):
		# load and resample audio data from 48 to 16kHz
		audio = batch["audio"]

		# compute log-Mel input features from input audio array 
		batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

		# encode target text to label ids 
		batch["labels"] = self.processor.tokenizer(batch["transcript"]).input_ids
		return batch
	
	def compute_metrics(self, pred):
		"""Method by Sanchit Gandi, from https://huggingface.co/blog/fine-tune-whisper#evaluation-metrics, adapted to use in FineTuner class"""
		metric = evaluate.load("wer")
		pred_ids = pred.predictions
		label_ids = pred.label_ids

		# replace -100 with the pad_token_id
		label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

		# we do not want to group tokens when computing the metrics
		pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
		label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

		wer = 100*metric.compute(predictions=pred_str, references=label_str) # note that line as written by Gandi is `wer = 100 * metric.compute(predictions=pred_str, references=label_str)` but this gives a warning of potential problem that the operator * isn't supported between these data types

		return {"wer": wer}

	def train(self):
		training_args = Seq2SeqTrainingArguments(
			output_dir="./fine-tuned-model",  # change to a repo name of your choice
			per_device_train_batch_size=16,
			gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
			learning_rate=1e-5,
			warmup_steps=500,
			max_steps=5000,
			gradient_checkpointing=True,
			fp16=True,
			per_device_eval_batch_size=8,
			predict_with_generate=True,
			generation_max_length=225,
			save_steps=1000,
			eval_steps=1000,
			logging_steps=25,
			report_to=["tensorboard"],
			load_best_model_at_end=True,
			metric_for_best_model="wer",
			greater_is_better=False,
			push_to_hub=False,
			)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	"""Data Collator class by Sanchit Gandi, from https://huggingface.co/blog/fine-tune-whisper#define-a-data-collator"""
	processor: Any
	decoder_start_token_id: int

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
		# first treat the audio inputs by simply returning torch tensors
		input_features = [{"input_features": feature["input_features"]} for feature in features]
		batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

		# get the tokenized label sequences
		label_features = [{"input_ids": feature["labels"]} for feature in features]
		# pad the labels to max length
		labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		# if bos token is appended in previous tokenization step,
		# cut bos token here as it's append later anyways
		if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
			labels = labels[:, 1:]

		batch["labels"] = labels

		return batch


def main():	
	finetuner = FineTuner()
	dataset = finetuner.make_dataset("res/validated-audio/metadata-subset.json")
	dataset_sample_test = dataset # TODO 2026/07/20 only attempting to use this for the purposes of testing that the trainer can instantiate correctly
	model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

	data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=finetuner.processor,decoder_start_token_id=model.config.decoder_start_token_id,)

	training_args = Seq2SeqTrainingArguments(
			output_dir="./fine-tuned-model",  # change to a repo name of your choice
			per_device_train_batch_size=16,
			gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
			learning_rate=1e-5,
			warmup_steps=500,
			max_steps=5000,
			gradient_checkpointing=True,
			fp16=True,
			eval_strategy="no", # Gandi guide says to set this to "steps", setting to 'no' while not passing an `eval_dataset` as per: ```ValueError: You have set `args.eval_strategy` to IntervalStrategy.STEPS but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. ```
			save_strategy='best', # as per ```ValueError: --load_best_model_at_end requires the save and eval strategy to match, except when --save_strategy="best", but found 	- Evaluation strategy: IntervalStrategy.NO 	- Save strategy: SaveStrategy.STEPS```
			per_device_eval_batch_size=8,
			predict_with_generate=True,
			generation_max_length=225,
			save_steps=1000,
			eval_steps=1000,
			logging_steps=25,
			report_to=["tensorboard"],
			load_best_model_at_end=True,
			metric_for_best_model="wer",
			greater_is_better=False,
			push_to_hub=False,
			)

	trainer = Seq2SeqTrainer(
		args=training_args,
		model=model,
		data_collator=data_collator,
		train_dataset=dataset, #training dataset
		#eval_dataset=dataset_sample_test, #testing dataset
		compute_metrics=finetuner.compute_metrics
	)

	trainer.train()
	trainer.save_model("fine-tuned-model")

if __name__ == "__main__":
	main()