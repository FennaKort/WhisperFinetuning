import glob
import json
import os
import torch
import whisper
from datetime import date, datetime
import logging

SUPPORTED_AUDIO_TYPES:list[str] = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'] 
"""audio filetypes supported by FFmpeg, which Whisper relies on for audio processing"""

ALL_MODELS:list[str] = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo'] 
"""A list containing the names of all Whisper models.""" #TODO 2026/06/24 planning to use this as a way to print and select models for use

class Transcriber:
	"""
	Class for transcribing speech in audio files using Whisper speech recognition models. 
	
	Initializes with defaults: audio directory 'res/audio/', transcript output directory 'res/transcriptions/, and Whisper model 'base.en'.
	"""

	def __init__(self, audio_dir:str="res/audio/", output_dir:str="res/transcriptions/", model_names:list[str]=["base.en"]) -> None:
		"""
		Args:
			audio_dir: the relative file path to the location to pull audio files from.
			output_dir: the relative file path the location to store completed transcripts in.
			model_names: a list of the names of the desired Whisper models to use for transcription. select from 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo'.
		"""
		# TODO 2026/06/24 may want to alternatively or additionally store multiple audio directory locations, or be able to store a list of absolute file paths for audio files?
		# TODO 2026/06/24 add behaviour to check for audio_dir and output_dir existence and to make dir if it doesn't exist

		self.logger = logging.getLogger("transcribe")
		logging.basicConfig(filename='transcribe.log', encoding='utf-8', level=logging.INFO)

		self.audio_dir = audio_dir
		self.output_dir = output_dir
		self.model_names = model_names
		self.device = ('cuda' if torch.cuda.is_available() else 'cpu')


	def set_audio_dir(self, audio_dir:str) -> None:
		"""Set the relative path of the directory location to pull audio files from."""
		# TODO add behaviour to check for audio_dir existence and to make dir if it doesn't exist
		self.audio_dir = audio_dir

	def get_audio_dir(self) -> str:
		"""Return the relative path to the location currently set to pull audio files from."""
		return self.audio_dir
	
	def set_output_dir(self, output_dir:str) -> None:
		"""Set the relative path of the directory location to store completed transcripts in."""
		# TODO add behaviour to check for output_dir existence and to make dir if it doesn't exist
		self.output_dir = output_dir
	
	def get_output_dir(self) -> str:
		"""Return the relative path to the location currently set to store completed transcripts in."""
		return self.output_dir

	def set_models(self, model_names:list[str]) -> None:
		# TODO 2026/06/24 want way to prompt to input model name, or type list to list all available models. can then select "all"
		"""
		Set the list of Whisper speech recognition models to use for transcription. 
		
		Select from 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo'."""
		self.model_names = model_names

	def get_model_names(self) -> list[str]:
		"""Return the list of Whisper speech recognition models currently selected for use in transcription."""
		return self.model_names
	
	def setup_audio_files(self) -> list[str]:
		"""
		Loads all supported audio files from the directory stored in self.audio_dir into a list.

		Whisper relies on FFmpeg for audio processing. Audio files must be in the following formats supported by FFmpeg: 'flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'.

		returns: 
			audio_files: list of all audio files found in specified location, with each file in the format 'audio_dir/file_name.file_type'
		"""
		# alogger = logging.getLogger("audio")
		# start_time = datetime.now()

		audio_files:list[str] = []
		# TODO add message if no audio files found and message confirming number of files found

		for file_type in SUPPORTED_AUDIO_TYPES:
			audio_files += glob.glob(self.audio_dir+"*."+file_type)
		audio_files.sort() # sorts to ensure list is alphabetical irrespective of file type 
		# alogger.info(f"Audio load duration: {datetime.now()-start_time}\n")

		return audio_files

	def transcription_controller(self):
		# contains prompts and printouts???
		pass

	def transcribe(self, audio_files:list[str], model_name:str) -> list:
		# confirm audio files to transcribe and model/models to use, prompt to continue or change locations and models
		"""
        Accepts a list of audio files and transcribes them using the Whisper model. 
		
		Choose from the following pre-trained models, or specify a fine-tuned model: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo'.

        Args:
            audio_files: list of audio files to transcribe; use setup_audio_files() to load a list of audio files from a target directory 
            model_name: the name of a Whisper model to use for transcription
            testing: True to run transcription on first audio file in list only, False to run transcription on all audio files in list
		Returns:
			transcripts: a list containing the model_name, number of files transcribed, and self.audio_dir in indexes 0-2, and dictionaries composed of each audio filename and resulting transcription after that.
        """
		# TODO 2026/07/01 rewrite docstring to clarify what speech segments data is stored

		tlogger = logging.getLogger("transcription")
		
		# start indicator:
		batch_start_time = datetime.now()
		tlogger.info(f"Transcribing {len(audio_files)} audio files with {model_name} model.")
		tlogger.info(f"Batch start time: {batch_start_time.isoformat(sep=" ", timespec="seconds")}")
		print(f"Transcribing {len(audio_files)} audio files with {model_name} model.")

		#load Whisper model:
		model = whisper.load_model(model_name).to(self.device)

		# set torch dtype decoding option to pass to Whisper transcriber:
		isFP16:bool = True # by default, use torch.float16
		if self.device == "cpu": # if on CPU, FP16 dtype is not available;
			isFP16 = False # allow Whisper transcriber to internally set dtype to FP32/torch.float32 instead

		transcripts:list = []

		for audio_file in audio_files:
			start_time = datetime.now() # file start
			result:dict = model.transcribe(audio = audio_file, word_timestamps=True, fp16=isFP16) #whisper returns dict containing fields "text","segments", "language"; we need the text field and some items from segments
			# RESOLVED TODO 2026-08-09: `decode-options` maybe set dtype according to self.device to avoid FP16 unsupported warning 

			segments:dict = result["segments"]

			# remove unneeded items from each segment:
			for segment in segments:
				segment.pop("seek")
				segment.pop("tokens")
				segment.pop("temperature")
				segment.pop("avg_logprob")
				segment.pop("compression_ratio")
				segment.pop("no_speech_prob")
			
			# find the end of the speech in the audio file; useful for segmenting audio for fine tuning operations
			last_segment:int = len(segments)-1 
			speech_end:float = segments[last_segment]["end"]

			transcript:dict = {"file_name": audio_file, "speech_ends_at": speech_end, "model_name": model_name, "manually_verified":False, "transcript": result["text"], "segments": segments} #"manually_verified" refers to whether transcript has been manually corrected for any transcription errors, False==no. 2026/06/28 including this for potential usefulness in metadata output

			transcripts.append(transcript)
			end_time = datetime.now() # file end
			tlogger.info(f"\t{audio_file} transcription time: {end_time-start_time}")

		# batch end indicator
		batch_end_time = datetime.now()
		tlogger.info(f"Batch end time: {batch_end_time.isoformat(sep=" ", timespec="seconds")}")
		tlogger.info(f"Total transcription time for batch: {batch_end_time-batch_start_time}\n")

		return transcripts

	def transcribe_with_multiple_models(self, audio_files:list[str]):
		for model_name in self.model_names:
			self.transcribe(audio_files, model_name)

	def setup_output_file_name(self, transcription_type: str, model_name: str) -> str: 
		"""Creates a string in the format output/dir/date-batch-transcription-model-name for use in naming output files. Note that the string does NOT include any file format specifier."""
		# Setup model name for use in output file name
		model_name = model_name.replace(".","-") # if model is a *.en model, replace the "." with "-" for use in file name output

		# return output/dir/date-batch-transcription-model-name for use in naming output files
		return self.output_dir + date.today().isoformat() + "-" + transcription_type + "-" + model_name

	def output_as_txt(self, transcripts:list) -> None:
		# Setup output file
		file_name:str = self.setup_output_file_name("batch-transcription",transcripts[0]['model_name']) +".txt"

		output_file = open(file_name, "w", encoding="utf-8") # TODO 2026/06/24 currently any transcriptions will overwrite previous transcripts created on the same day using the same model. Would prefer to differentiate this without adding repeat transcripts to the file upon appending, think on this later

		# Save transcript(s) to output file:
		output_file.write("Model: " + transcripts[0]["model_name"] + "\nAudio file(s) transcribed: " + str(len(transcripts)) + "\nFrom location: " + self.audio_dir)

		for transcript in transcripts:
			output_file.write("\n\n" + "File: " + transcript["file_name"])
			output_file.write("\n" + "Transcript: " + transcript["transcript"])
		
		print(f"Transcription saved to: {file_name}")

	def output_as_json(self, transcripts:list) -> None:
		# Setup output file
		file_name:str = self.setup_output_file_name("batch-metadata",transcripts[0]['model_name']) +".json" #TODO 2026/07/01 may rework both output methods to allow for customization of the output filename? or maybe some way to specify whether you want to customize it within setup_output_file_name()?

		with open(file_name,'w', encoding='utf-8') as json_file:
			json.dump(transcripts,json_file, indent=4)

		print(f"Transcription metadata saved to: " + file_name)

	def test_transcriber(self) -> None:
		test_transcript = self.transcribe(["res/audio/voice-message-01.mp3","res/audio/voice-message-02.mp3"], 'tiny.en')

		self.output_as_txt(test_transcript)
		self.output_as_json(test_transcript)
	
	def batch_transcriber(self):
		"""Iterates through all models stored in self.model_names list to transcribe all audio files currently in the audio directory and output a separate transcription file per model used, as well as a single metadata.json file containing metadata for all transcriptions"""
		
		audio_files:list = self.setup_audio_files()

		for model_name in self.get_model_names():
			transcripts:list = self.transcribe(audio_files=audio_files, model_name=model_name)
			self.output_as_txt(transcripts)
			self.output_as_json(transcripts)

# want class for organizing finetuning data storage???

def main():
	transcriber = Transcriber(model_names=['tiny.en'])
	# transcriber.test_transcriber()

	transcriber.set_models(['tiny','tiny.en','base','base.en''small','small.en'])
	transcriber.batch_transcriber()

if __name__ == "__main__":
	main()