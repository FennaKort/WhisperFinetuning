import json
import os
import torch
import transcribe
import whisper

from pydub import AudioSegment # TODO 2026/07/03 not suuuure I need this? I may be able to accomplish splitting with ffmpeg-python instead? 

# TODO 2026/07/03: needs to:
# [] load unprocessed metadata
# [] check metadata for files that need to be chunked
# 	[] if less than 30s and doesn't need to be chunked, insert into checked metadata list
# 	[] if needs to be chunked:
# 		[] calculate split locations
# 		[] split Audio
# 		[] create new matching metadata while preserving relative timestamps (don't want to retranscribe ever)
# 		[] insert new metadata into checked metadata list		
# 		[] continue to next unchecked entry in unchecked metadata list
					


class DataEntry:
	def __init__(self, file_name: str, speech_ends_at: float, model_name: str, manually_verified: bool, transcript: str, segments: list) -> None:
		self.file_name = file_name
		self.speech_ends_at = speech_ends_at
		self.model_name = model_name
		self.manually_verified = manually_verified
		self.transcript = transcript
		self.segments = segments

class AudioProcessor:
	def __init__(self) -> None:
		self.metadata:list

	def get_metadata(self) -> list:
		return self.metadata
	
	def set_metadata(self, metadata: list) -> None:
		self.metadata = metadata
		
	def load_metadata_from_json(self, file_path: str) -> None:
		with open(file_path, 'r') as json_file:
			self.set_metadata(json.load(json_file))
	
	def print_metadata_details(self) -> None:
		if len(self.metadata) > 0:
			print(f'Metadata found for {len(self.metadata)} audio files.')

			#TODO 2026/07/03 good idea to update the following to avoid printing out a gajillion details lol
			for entry in self.metadata:
				print(entry['file_name'])
				print(f"First text segment: {entry['segments'][0]['text']}\n")

	def evaluate_metadata(self, metadata:list) -> list:
		validated_metadata:list = []
		for item in metadata:
			if item["speech_ends_at"] < 30.0:
				validated_metadata.append(item)
			else:
				print("Slicing file: " + item["file_name"])
				self.calculate_slice(item["segments"])
				# TODO 2026/07/03 continue this by continuing to add behaviour to complete slices and add to new validated meta data list
			

		return validated_metadata

	def calculate_slice_points(self, segments: list):
		"""
		Determine where to slice an audio file into chunks less than 30 seconds according to logical cutoff points within the audio's transcription.

		Args: 
			segments: a list of audio segment details; each segment includes fields "id", "start", "end", "text", and "words" (containing additional word-level details)
		"""
		slices:list = []
		
		slices.append(self.calculate_slice(segments))
		print(f"This audio file has {len(slices)} slices")

	def calculate_slice(self, segments: list, starting_segment:int = 0, chunk_end:float = 30.0) -> list:
		"""
		Determine where to slice an audio file into chunks less than 30 seconds according to logical cutoff points within the audio's transcription.

		Args: 
			segments: a list of audio segment details; each segment includes fields "id", "start", "end", "text", and "words" (containing additional word-level details)
		"""
		# base case: reach last segment in segments
		slice:list = []

		if starting_segment == 0:
			max_chunk_end:float = 30.0 # initializes audio chunk length counter to maximum chunk size of 30s
		else: 
			max_chunk_end:float = chunk_end

		# TODO 2026/07/03 next: figure out what I'm writing wrong here lol, currently the code continues checking the first method call with starting_segment = 0 but now with max_chunk_end = 58.76 after finishing the recursive call, leading to any segments that are supposed to be getting handled by the recursive call getting RE-handled
		
		x = starting_segment
		for x in range(starting_segment,len(segments)):
			if x == len(segments):
				break
			
			if segments[x]["end"] < max_chunk_end: # if segment is within chunk, add to slice
				print(f"{segments[x]["id"]}: " + segments[x]["text"])
				slice.append(segments[x])
				x+=1

			else: # else, start new slice
				print(f'new slice starts at: {segments[x]['start']}')
				max_chunk_end = (max_chunk_end + 30.0 - segments[x]['end'] + 30.0) # subtract actual end of final segment from max chunk end and add 30s to find new max chunk end
				print(max_chunk_end)
				self.calculate_slice(segments, x, max_chunk_end)
				return slice
		return slice	
	# start here and look until you've got a chunk, or until you've got the last segment


	
	def split_audio(self, audio_file_path:str, splits:list[dict])-> None:
		"""
		Args:
			audio_file_path: relative path to audio file
			split_at: the time to split the file at in seconds
		"""
		# if needs splitting (check if speech_ends_at property in json metadata is greater than 30s):
			# check number of segments
			# check segment duration using modular arithmetic? 
			# while segments are still left, check segment end time mod 30?

		# load audio from file
		# find length of audio
		# 


		# if beeg, split; actually maybe decide if it needs to be split outside of this method?
		# maybe can call recursively until no segments are too long??? this is behaviour to decide on outside this method
		# going to need to be able to preserve the segment text-duration relationships
		# could maybe 

		# load audio from file path using pydub AudioSegment
		audio:AudioSegment = AudioSegment.from_file(audio_file_path)
		
		duration:int = len(audio) # maybe not needed??

		# Iterate through the list of timestamps to split the audio at
		for split in splits: # list of dicts containing start and end times for segments

			start_milliseconds = split['start'] * 1000 # convert start time from seconds to milliseconds

			end_milliseconds = split['end'] * 1000 # convert end time from seconds to milliseconds



			chunk = audio[start_milliseconds:end_milliseconds] 



			#output_file = 


def main() -> None:
	audio_processor = AudioProcessor()
	audio_processor.load_metadata_from_json('res/transcriptions/2026-07-01-batch-transcription-metadata.json')
	audio_processor.print_metadata_details()
	audio_processor.evaluate_metadata(audio_processor.get_metadata())

	
if __name__ == "__main__":
	main()