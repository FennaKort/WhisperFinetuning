import glob
import os
import torch
import whisper
from datetime import date
from pathlib import Path


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

class WhisperBatchTranscriber:
    def __init__(self) -> None:
        pass
    # audio file path and name setup
    script_dir = os.path.dirname(os.path.abspath(__file__)) # absolute path of current python file to sidestep working directory issue I was having b/w running in powershell (where it worked without this) vs vscodium (where it didn't lol)
    # note 2026/06/22: this was copied in from a previous transcription test; figure out how to reimplement

    # audio_dir = "./res/audio/"
    # audio_file_name = "voice-message-4.ogg" 
    # audio_file = script_dir + audio_dir + audio_file_name

    # output_dir = "./res/transcriptions/"

    # Whisper model name setup
    # model_name = "tiny.en"
    # model_output_name = model_name # used to append the model name to the file name upon output
    # model_output_name = model_name.replace(".","-") # if model is a *.en model, replace the "." with "-" for use in file name output

    # Load Whisper model
    # model = whisper.load_model(model_name)

    def setup_audio_files(self,audio_dir:str) -> list:
        """
        Loads all supported audio files from the specified directory into a list. Audio files must be in the following formats supported by FFmpeg, which Whisper relies on for audio processing: 'flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'

        Args: 
            audio_dir: the relative path to the location your audio files are stored in (ie., "res/audio")
        Returns: 
            audio_files: list of all audio files found in specified location
        """
        supported_filetypes:list[str] = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'] # audio filetypes supported by FFmpeg, which Whisper relies on for audio processing
        audio_files:list[str] = []

        # TODO add behaviour to check for audio_dir existence and to make dir if it doesn't exist
        # TODO add message if no audio files found and message confirming number of files found

        for file_type in supported_filetypes:
            audio_files += glob.glob(audio_dir+"*."+file_type)
        print(audio_files)
        return audio_files

    def transcribe_audio_files(self, audio_files:list[str], model_name:str) -> str:
        """
        Accepts a list of audio files and transcribes them using the specified Whisper model. Choose from the following pre-trained models, or specify a fine-tuned model: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo'.

        Args:
            audio_files: list of audio files to transcribe; use setup_audio_files() to load a list of audio files from a target directory 
            model_name: the name of a Whisper model to use for transcription
            testing: True to run transcription on first audio file in list only, False to run transcription on all audio files in list
        """
        model = whisper.load_model(model_name) # loads the specified Whisper model
        # TODO may rework model_output_name depending on whether or not I continue to use/offer text file output?
        model_output_name = model_name # used to append the model name to the file name upon output
        model_output_name = model_name.replace(".","-") # if model is a *.en model, replace the "." with "-" for use in file name output

    # TODO rework transcription output later
        # transcription_info:dict = {"audio_directory":audio_dir} # TODO this line needs to be inside a different function in order to accept audio directory info
        transcription:str = "Model: " + model_name + "\n" + str(len(audio_files)) + " audio file(s) transcribed. \n" # TODO would need to reconfigure to have access to audio_dir in this function if I wanted to output audio dir location
        # run transcription on all audio files in list
        for audio_file in audio_files:
            print(audio_file)
            result:dict = model.transcribe(audio_file)
            transcription += "\nFile: " + str(audio_file) + "\nTranscript:" + result["text"] +"\n"
        
        output_dir = "res/transcriptions/"
        # TODO want: audio_dir name, name of model used, number of audio files. list of audio_file-transcript pairings.
        
        # Save transcription to a file
        text_file = output_dir + date.today().isoformat() + "-batch-transcription-"+ model_output_name +".txt"
        with open(text_file, "w", encoding="utf-8") as file:
            file.write(transcription)

        print(f"Transcription saved to: {text_file}")

        return transcription


    # TODO move output functionality to this function?? 
    def output_transcripts(self, output_dir:str, transcription:str): # TODO would be helpful to have model_name as key in dict instead I think
        output_dir = "res/transcriptions"
        # TODO want: audio_dir name, name of model used, number of audio files. list of audio_file-transcript pairings.
        
        # Save transcription to a file
        text_file = output_dir + date.today().isoformat() + "-batch-transcription-"+ model_output_name +".txt"
        with open(text_file, "w", encoding="utf-8") as file:
            file.write(transcription)

        print(f"Transcription saved to: {text_file}")
        print("\n Transcription:\n", transcription)

def main():
    transcriber = WhisperBatchTranscriber()

    # single file transcription for testing:
    transcriber.transcribe_audio_files(["res/audio/voice-message-1.mp3"],"tiny.en")

    # transcribe all files in dir:
    # audio_files:list = transcriber.setup_audio_files("res/audio/")
    # #TODO figure out if it's better to load models separately or within transcribe_audio_files function?
    # models_smaller:list = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en']
    # models_larger:list = ['medium', 'medium.en', 'large', 'turbo']
    # models:list = models_smaller + models_larger

    # for model in models:
    #     transcriber.transcribe_audio_files(audio_files,model)

main()