from datamodel import *
import json
import os

from pydub import AudioSegment # TODO 2026/07/03 not suuuure I need this? I may be able to accomplish splitting with ffmpeg-python instead? 

# TODO 2026/07/03: needs to:
# [x] load unprocessed metadata
# [x] check metadata for files that need to be chunked
# 	[x] if less than 30s and doesn't need to be chunked, insert into checked metadata list
# 	[x] if needs to be chunked:
# 		[x] calculate split locations
# 		[x] split Audio
# 		[x] create new matching metadata while preserving relative timestamps (don't want to retranscribe ever)
# 		[x] insert new metadata into checked metadata list		
# 		[x] continue to next unchecked entry in unchecked metadata list
# [] output validated metadata to json file
					
class DataProcessor:
	def __init__(self) -> None:
		self.metadata:list
		self.validated_audio_dir = "res/validated-audio/" # TODO 2026-07-08 need to add behaviour for checking existence of dirs

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
		#TODO 2026/07/08 maybe I want to only be evaluating metadata that's marked as manually_verified == True, consider including later

		for item in metadata:
			if item["speech_ends_at"] < 30.0:
				validated_metadata.append(item)
			else:
				print("Slicing file: " + item["file_name"])
				entry:DataEntry = DataEntry(file_name=item["file_name"], speech_ends_at=item["speech_ends_at"], model_name=item["model_name"], manually_verified=item["manually_verified"], transcript=item["transcript"], segments=item["segments"])

				new_metadata:list = self.slice_item(entry)
				validated_metadata.extend(new_metadata) # TODO 2026/07/08 figure where I want to convert from json str/list items to DataEntry objects: do I want to convert all items in metadata to data entry objects upon loading? or do I only want to convert the items that are necessary for updating the metadata, and then convert back before writing to the validated metadata list?
			

				# TODO 2026/07/03 continue this by continuing to add behaviour to complete slices and add to new validated meta data list
		print(len(validated_metadata))

		return validated_metadata

	def slice_item(self, entry: DataEntry) -> list[DataEntry]:
		"""
		Handle the slicing of an item into new metadata and matching new audio files

		Returns:
			new_metadata: a list of new metadata entries as DataEntry objects for the new split audio files
		"""
		new_metadata:list[DataEntry] = [] # list of new metadata entries
		model_name: str = entry.model_name
		manually_verified: bool = entry.manually_verified
		
		# 1. find slices
		slices:list[list[Segment]] = self.find_slices(entry.segments)

		# 2. use slice locations to slice audio files
		new_file_names: list = self.split_audio(entry.file_name, slices)

		print('slice_item:')
		for file in new_file_names:
			print(file)

		# 3. use slices to reconstruct new metadata
		for i in range(0, len(slices)): # for each slice:
			speech_end: float = slices[i][-1].end
			transcript:str = ""
			for segment in slices[i]:
				transcript += segment.text
			new_entry:DataEntry = DataEntry(file_name=new_file_names[i], speech_ends_at=speech_end, model_name=model_name, manually_verified=manually_verified, transcript=transcript, segments=slices[i])
			new_metadata.append(new_entry)

		return new_metadata

	def find_slices(self, segments:list) -> list[list[Segment]]:
		"""
		Determine where to slice an audio file into chunks less than 30 seconds according to grammatical sentence cutoff points within the audio's transcription. 
		
		All segments in the first 15.0 seconds of a chunk are automatically included in a slice. Segments in the last 15.0 seconds are included if their "text" field ends on one of "!", "?", ".", "。", "！" (U+ff01), or "？" (U+ff1f), indicating the end of a sentence. If their text field ends on another character, the current slice is ended and the segment is added to the beginning of a new slice.

		Args:
			segments: a list of audio segment transcription details; each segment includes fields "id", "start", "end", "text", and "words" (containing additional word-level details). 
		Returns:
			slices: a list of groupings of segments; each slice is delineated by the segment with the last grammatical sentence end that occurs within a potential 30s audio chunk. the "end" value of the final segment in each slice is a point at which the audio file should be split. the metadata for the original audio file should be reconstructed by making one new entry per slice and associating it with the matching split audio file. 
		"""

		max_chunk_end: float = 30.0 # initializes audio chunk length counter to maximum Whisper audio chunk size of 30s when dealing with first segment in list of segments
		slice:list[Segment] = [] # a single slice is a group of transcript segments ending with the final segment that has a valid sentence end within the given max_chunk_end
		slices:list[list[Segment]] = [] # a list of all the slices that should be constructed from the given list of segments

		slice_length_counter: float = 0.0 # keeps track of the current end of the last segment in the slice, is used for checking grammatical ends of sentences when near the end of max chunk
		slice_counter:int = 0 # TODO slice_counter var is being used for debugging slice turnover behaviour, can likely remove in the future

		print('new slice starts at: 00:00')

		for i in segments:
			segment: Segment = Segment(id=i['id'], start=i['start'], end=i['end'], text=i['text'], words=i['words'])

			if segment.end < max_chunk_end:
				if slice_length_counter < (max_chunk_end - 15.0):
					slice.append(segment)
					print(f"{segment.id}: " + segment.text)
					slice_length_counter = segment.end
				else: # if segment is nearing the end of the max chunk, check if it is the end of a sentence
					if segment.text[-1] in ["!", "?", ".", "。", "！", "？"]:
						# if sentence end, add segment to slice
						slice.append(segment)
						print(f"{segment.id}: " + segment.text)
						slice_length_counter = segment.end
					# for segments nearing the end of the current max chunk and do not end on a grammatical sentence break:
					else: # if NOT sentence end, end current slice and add current segment to new slice
						slices.append(slice) # add current slice to list of slices
						slice = [] # reassigned the list reference for slice to start a new slice
						
						slice_counter += 1 
						print(f'slice {slice_counter} end \n')
						
						slice_length_counter = 0.0 # reassign the counter to reset
						max_chunk_end = segment.start + 30.0 

						print(f'new slice starts at: {segment.start}')
						slice.append(segment) # new slice starts with current segment as first segment of slice
						print(f"{segment.id}: " + segment.text)
						slice_length_counter = segment.end
			else:
				slices.append(slice) # add current slice to list of slices
				slice = [] # reassigned the list reference for slice to start a new slice

				slice_counter += 1
				print(f'slice {slice_counter} end \n')

				slice_length_counter = 0.0 # reassign the counter to reset
				max_chunk_end = segment.start + 30.0

				print(f'new slice starts at: {segment.start}')
				slice.append(segment) # new slice starts with current segment as first segment of slice
				print(f"{segment.id}: " + segment.text)
				slice_length_counter = segment.end 

		slices.append(slice) # add final slice to list of slices
		slice_counter += 1
		print(f'slice {slice_counter} end \n')

		return slices
	
	def split_audio(self, audio_file_path:str, slices:list[list[Segment]])-> list:
		"""
		Splits an audio file into multiple parts determined by the start and end points of each slice in the slices list param. 
		
		The slices list should be prepared using the find_slices() method. This audio splitting method is adapted from https://codesignal.com/learn/courses/transcribing-large-files-in-python-using-pydub/lessons/splitting-large-audio-files-with-pydub-for-efficient-transcription

		Args:
			audio_file_path: relative path to audio file
			slices: list of transcript slices for the audio file (a single slice being a list of Segment objects)
		Returns:
			new_file_paths: list of file paths for new audio files created by splitting operation
		"""
		
		new_file_paths: list = [] # list that will be returned storing file paths for the new audio files
		slice_counter: int = 1 # counts which slice of the original audio we're currently operating on

		# Iterate through the list of locations to split the audio at
		for slice in slices: # each slice stores a list of Segment objects created with find_slices(), start time of first Segment and end time of last Segment determine split locations
			if slice[0].id == 0:
				start_milliseconds = 0.0 # if first segment in slice is at start of audio file, set start in milliseconds to 0
			else:
				start_milliseconds = slice[0].start * 1000 # convert start time of first segment in slice from seconds to milliseconds
			end_milliseconds = slice[-1].end * 1000 # convert end time of last segment in slice from seconds to milliseconds
			
			# Setting up output file name parts
			file_parts:tuple[str,str] = os.path.split(audio_file_path) # separates into [head=audio_dir/, tail=file_name.file_extension]

			original_file_name = os.path.splitext(file_parts[1])[0] # store file_name from tail
			file_ext = os.path.splitext(file_parts[1])[1] # store file_extension from tail

			# output_file path will be validated_audio_dir/original_file_name-split-slice_counter.file_extension
			output_file:str = self.validated_audio_dir + original_file_name + f'-split-{slice_counter}{file_ext}'
			print(output_file)
			
			# load audio from file path using pydub AudioSegment
			audio = AudioSegment.from_file(audio_file_path)

			# Split audio chunk at slice point
			print(f"Extracting slice {slice_counter}")
			slice_counter += 1

			chunk = audio[start_milliseconds:end_milliseconds]
			chunk.export(output_file, format=file_ext.lstrip('.'))
			new_file_paths.append(output_file)
		
		print(f"Split {audio_file_path} into {len(new_file_paths)} chunk(s):")
		return new_file_paths

def main() -> None:
	data_processor = DataProcessor()
	
	# to test data processing on a subsection of metadata:
	data_processor.load_metadata_from_json('res/transcriptions/2026-07-03-batch-transcription-metadata.json') # contains tiny.en model transcripts for two files in audio dir
	data_processor.print_metadata_details()

	# to test data processing on metadata for all audio files in audio dir:
	# data_processor.load_metadata_from_json('res/transcriptions/2026-07-08-metadata-tiny-en-subset.json') #2026-07-08 manually created subset of metadata from "res\transcriptions\2026-07-08-batch-transcription-metadata.json" containing only transcripts from tiny.en model

	data_processor.evaluate_metadata(data_processor.get_metadata())

	
if __name__ == "__main__":
	main()