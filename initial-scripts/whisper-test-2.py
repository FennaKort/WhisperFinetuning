# following method from here: https://medium.com/@selinamangaroo/an-introduction-to-whisper-speech-to-text-transcription-a3b86429c59a

import whisper
import os

# audio file path and name setup
script_dir = os.path.dirname(os.path.abspath(__file__)) # absolute path of current python file to sidestep working directory issue I was having b/w running in powershell (where it worked without this) vs vscodium (where it didn't lol)

audio_dir = "./res/audio/"
audio_file_name = "voice-message-4.ogg" 
audio_file = script_dir + audio_dir + audio_file_name

output_dir = "./res/transcriptions/"

# Whisper model name setup
model_name = "tiny.en"
model_output_name = model_name # used to append the model name to the file name upon output
model_output_name = model_name.replace(".","-") # if model is a *.en model, replace the "." with "-" for use in file name output

# Load Whisper model
model = whisper.load_model(model_name)

# Transcribe the audio
print("Transcribing...")
try:
    result = model.transcribe(audio_file) # yeah so the issue is still with that call to the python subprocess from within something in transcribe?
    transcription = result["text"]

    # Save transcription to a file
    text_file = output_dir + os.path.splitext(audio_file_name)[0] + "_transcription_"+ model_output_name +".txt"
    with open(text_file, "w", encoding="utf-8") as file:
        file.write(transcription) # type: ignore

    print(f"Transcription saved to: {text_file}")
    print("\n Transcription:\n", transcription)

except Exception as e:
    print(f"Error: {e}")

# ok yeah so I'm still running into the same issue of some subprocess thing not working. should i try to figure that out? or should i try setting up a virtual environment and seeing if that fixes it? //2026/06/10 ETA that I did set up a venv and this subprocess aspect IS currently working, but I don't recall if I wrote down WHAT fixed it T.T