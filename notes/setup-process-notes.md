check system for cuda compatibility and cuda availability, direct user to install cuda if compatible but unavailable?

can run `winget install ffmpeg` automatically??? to install ffmpeg first because it auto-installs and automatically adds to path.
for cpu only, run `py -m pip install torch torchaudio ffmpeg-python openai-whisper`. Intel integrated gpu acceleration is seemingly possible for RUNNING whisper but not for finetuning it from what I'm reading? NOT SURE. but [this is a library full of intel optimizations for pytorch](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-pytorch.html), so maybe something here?
	- and [here's a whisper tutorial for running on intel cpu using these optimizations i think](https://www.intel.com/content/www/us/en/developer/articles/technical/speech-recognition-in-openai-whisper-without-a-gpu.html)

to run debugger using virtual environment in vscode, need to include `"justMyCode": false //necessary for VSCode to be able to launch virtual environment when using debugger` in `launch.json` debug configuration file

to set default venv use in vscode, click into python file, have python and python environments extensions installed, click python interpreter button at bottom right of status bar (to the right of the Python language indicator), select the working directory you intend to use, and then select the correct python interpreter for that directory


need to write wer calculation
need to manually correct sample transcriptions
need to figure out where to put WER calc script
[x] need to organize repository for publishing



------
# For Finetuning: 
## FIRST, need to output data in a useful JSON format and splitting audio and segmenting
- 2026-07-16 note: I have ran these segmenting on the 10 sample files using both tiny.en and small.en transcripts, and I've found that the splitting from the small.en segments is actually much worse/less useful for my purposes than from tiny.en. I'd be curious to find some way to quantify this. In general, so far small.en seems to make less sentence ends, and end the segments on sentence ends less frequently than tiny.en. This results in less useful sentence breaks. I'm also noticing that this is occasionally resulting in audio being split too early and splitting the middle of a word. It would be very interesting to compare the usefulness of training data generated from each of these types of segmenting. 
- If it turns out that segmenting from tiny.en is more useful than the segmenting from small.en, is there some way to combine the transcripts from small.en with the segmenting from tiny.en in order to give the and use are the best of both (ie., the more accurate out of the box transcripts from small.en and the improved segments from tiny.en)
- Another approach might be to have separate segmenting strategies for both models, for example using word-level metadata for slicing and splitting these segments of small.en transcripts but continuing to use the segment-level details if the model was originally tiny.en
- I would like to perhaps build everything from word-level details as that might be an easier way to have individuals correct words and store the manually verified transcripts? Finding out the best process to store manually verified transcripts on an as-created basis is a future matter
- Ok but for now, the most important thing is to move on to setting up the fine-tuning pathway

## SECOND, need to create custom dataset in huggingface format:
As per [hugging face directions on creating custom dataset](https://huggingface.co/docs/datasets/audio_dataset)
- [convert audio dataset to huggingface format and then drop in custom dataset for `common_voice` dataset in main huggingface finetuning blog post](https://huggingface.co/spaces/openai/whisper/discussions/75) 
- I think I actually need to work in python dictionary form FIRST in order to easily be storing the info correctly because as per the [Local files](https://huggingface.co/docs/datasets/audio_dataset#local-files) subheading, you can use Dataset.from_dict(<dictionary>.cast_column("audio", Audio())) to load from a dictionary and cast the audio file paths to the transformers Audio feature
- "Here the file_name must be the name of the audio file next to the metadata file. More generally, it must be the relative path from the directory containing the metadata to the audio file." Ok so I actually need to separate out the relative path and the you files individual name in the JSON 
- okay it sounds like you need to have a dictionary of the audio files AND a metadata.csv/metadata.json with `file_name,additional_feature` labels and can't use the same output for both???
- I WANT to be able to store a .json file with the audio paths and validated transcripts. would be great if every time the audio is marked as usable for fine-tuning, it's stored somewhere specific? 
- or maybe because I'll want to be able to store the audio in the same spot as the `metadata` file, I want my `Transcriber` to ask to select files from the location rather than select ALL files from the location??? what do I want to do about duplicates of the same audio file? THIS IS A LATER LIST TASK
- what I want in the metadata is 
- hmmmm based on the description of what columns to keep vs drop in [this finetuning article](https://www.graphcore.ai/posts/fine-tune-openais-whisper-automatic-speech-recognition-asr-model), I think the json can store `file_name,transcription` and then I can load that to a dictionary that converts `file_name` to an `audio` field in the dictionary and then cast THAT to the Audio type


### [article on loading local custom datasets using `load_dataset`](https://huggingface.co/docs/datasets/audio_load)
- must have a `metadata.csv` file or a `metadata.json` file in the `AudioFolder`

### NOT USING [processing audio using Zero Crossing Rate and energy level to detect silences for segmenting](https://medium.com/@lenny.bijan/finetuning-openais-whisper-creating-your-custom-dataset-i-a6e7a5894a2d)
#### how to preprocess your audio
- any audio beyond 30s isn't automatically truncated to 30s, it's outright disregarded, so we ALWAYS need to segment the audio effectively prior to fine-tuning (this appears to be incorrect according to the huggingface article, it's just that it doesn't truncate it at useful spots automatically)
- "Human speech, when broken down into very short segments such as 30 milliseconds (ms), can generally be considered as static [1]. During such a brief interval, the vocal apparatus (larynx, tongue, lips, etc.) does not have the physical capability to produce dramatic changes in sound. This characteristic is crucial for developing effective audio preprocessing strategies because it provides a foundation for making assumptions about the continuity and stability of speech within these short windows."
- "we will utilize two sophisticated metrics: the Zero Crossing Rate (ZCR) and the energy level of the audio. Instead of depending on a fixed decibel threshold, we’ll dynamically calculate the mean ZCR and energy for each individual audio file. To enhance precision, we will set our thresholds at the mean plus one standard deviation of these measurements. In practice, for an audio frame to be classified as silent, it must fall below both the ZCR and energy thresholds. This dual-check ensures that we only deem a segment silent when it truly exhibits low levels of both vocal activity and sound intensity, thus minimizing the risk of erroneously discarding important audio content [2]."
- actually realizing that this method of audio processing isn't going to be easily compatible with pre-existing transcripts

### [processing audio using word-level timestamps](https://www.youtube.com/watch?app=desktop&v=OfQNgPfv97s)
- 13:08 use whisper-timestamped model vs regular whisper with timestamps - regular whisper will give timestamps with irregular lengths (but it seems like whisper-timestamped will too? still unclear as to why he uses this rather than regular whisper with word-level timestamps)
- 14:50 don't want segments that start in the middle of a sentence because that's harder for the model to understand the whole-sentence context
- 15:05 use word-level timestamps to assemble your own segments with a controlled length. ?? curious why he uses whisper-timestamped and not just regular whisper with word-level timestamps??
	- [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) uses a different implementation of timestamping than regular whisper
- 20:00ish looks like he's not actually going to walk us through the segment construction process lol but it might be feasible to construct from what's shown on screen
- sounds like [the Whisper feature extractor automatically pads to 30s](https://huggingface.co/blog/fine-tune-whisper#load-whisperfeatureextractor), so we don't need to pad. it would also auto-truncate to 30s, but not on useful stopping points. We also need to convert all audio to 16kHz sampling rate before we pass it to the feature extractor.

### [splitting audio files using ffmpeg](https://readmedium.com/split-and-transcribe-audio-files-with-openai-whisper-cee0b89a509d):
- still may be interested in getting ffmpeg and/or python ffmpeg splitting working in the future in order to remove PyDub dependency
	//cmd = ["ffmpeg", "-ss", str(start_milliseconds), "-i", audio_file_path, "-t", str(length), "-c", "copy", output_file]
	//run(cmd, capture_output=True, check=True).stdout
	//ffmpeg -ss 00:00 -t 120 -i big_mp3.mp3 output.mp3




## method from https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/
additionally need to install:
`py -m pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio`
