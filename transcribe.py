import glob
import os
import torch
import whisper
from datetime import date

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

		self.audio_dir = audio_dir
		self.output_dir = output_dir
		self.model_names = model_names
		self.current_model = whisper.load_model(model_names[0]) #need to figure out if I should be loading the model within or outside of the transcribe action

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
		audio_files:list[str] = []
		# TODO add message if no audio files found and message confirming number of files found

		for file_type in SUPPORTED_AUDIO_TYPES:
			audio_files += glob.glob(self.audio_dir+"*."+file_type)
		print(audio_files)
		return audio_files

	def transcription_controller(self):
		# contains prompts and printouts???
		pass

	def transcribe(self, audio_files:list[str], model_name:str) -> list[str]:
		# confirm audio files to transcribe and model/models to use, prompt to continue or change locations and models
		"""
        Accepts a list of audio files and transcribes them using the Whisper model. 
		
		Choose from the following pre-trained models, or specify a fine-tuned model: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo'.

        Args:
            audio_files: list of audio files to transcribe; use setup_audio_files() to load a list of audio files from a target directory 
            model_name: the name of a Whisper model to use for transcription
            testing: True to run transcription on first audio file in list only, False to run transcription on all audio files in list
		Returns:
			transcription: a list[str] containing the model_name, number of files transcribed, and self.audio_dir in indexes 0-2, and alternating audio filenames and resulting transcriptions after that.
        """
		model = whisper.load_model(model_name) # loads the specified Whisper model

		transcription:list[str] = [model_name, str(len(audio_files)), self.audio_dir]
		# TODO 2026/06/24 want to convert to use JSON instead but currently focusing on replicating behaviour for text output

		for audio_file in audio_files:
			result:dict = model.transcribe(audio_file)

			transcription.append("File: " + str(audio_file))
			transcription.append("Transcript: " + result["text"])
			# TODO 2026/06/24 want to convert to use JSON instead but currently focusing on replicating behaviour for text output

		return transcription

	def transcribe_with_multiple_models(self, audio_files:list[str]):
		for model_name in self.model_names:
			self.transcribe(audio_files, model_name)

	def output_transcripts(self, transcripts:list[str]) -> None:
		# TODO 2026/06/24 move descriptors to an output formatting function
			# ["Model: " + model_name, "Audio file(s) transcribed: " + str(len(audio_files)), "From location: " + self.audio_dir]
		
		# Setup model name for use in file name
		model_output_name = transcripts[0] # used to append the model name to the file name upon output
		model_output_name = model_output_name.replace(".","-") # if model is a *.en model, replace the "." with "-" for use in file name output
		
		# Setup output file
		text_file:str = self.output_dir + date.today().isoformat() + "-batch-transcription-"+ model_output_name +".txt"
		
		output_file = open(text_file, "w", encoding="utf-8") # TODO 2026/06/24 currently any transcriptions will overwrite previous transcripts created on the same day using the same model. Would prefer to differentiate this without adding repeat transcripts to the file upon appending, think on this later

		# Save transcript(s) to output file:
		output_file.write("Model: " + transcripts[0] + "\nAudio file(s) transcribed: " + transcripts[1] + "\nFrom location: " + self.audio_dir)

		i = 3
		while i<len(transcripts):
			if i%2 != 0:	
				output_file.write("\n")
			output_file.write("\n"+transcripts[i])
			# add new line between each filename-transcript pair
			i+=1
		
		print(f"Transcription saved to: {text_file}")
	
	def test_transcriber(self) -> None:
		test_transcript = self.transcribe(["res/audio/voice-message-1.mp3", "res/audio/voice-message-2.mp3"], 'tiny.en')

		self.output_transcripts(test_transcript)

# want class for organizing finetuning data storage???

def main():
	transcriber = Transcriber(model_names=['tiny.en'])

	# transcriber.test_transcriber()

	transcriber.set_models(['tiny','tiny.en','base','base.en','small','small.en'])
	for model_name in transcriber.get_model_names():
		transcriber.output_transcripts(transcriber.transcribe(audio_files=transcriber.setup_audio_files(), model_name=model_name))


	
main()