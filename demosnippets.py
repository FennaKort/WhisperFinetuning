from whisper import load_audio, pad_or_trim
from transformers import WhisperProcessor
from evaluate import load

def demo_load_audio_output():
	"""
	Demonstration of Whisper's load_audio() function showing how the size of the audio amplitude arrays returned by the function are affected by passing in different audio sampling rates. 
	
	Shows that the amplitude arrays are shorter when the function resamples audio at Whisper's native sampling rate of 16kHz, and that Whisper will automatically resample to this rate if no sampling rate is passed into load_audio(). Using this process therefore replicates the functionality of `datasets` `cast_column("audio", Audio(sampling_rate=16000))` method, which is the resampling approach recommended in most Whisper fine-tuning guides that use premade audio datasets that already have amplitude array data constructed from their audio files' native sampling rates. 
	"""
	audio_array = load_audio("res/audio/voice-message-1.mp3", sr=44100) #44.1kHz is a typical native sampling rate for audio, so I am using sr (samplingrate) = 44100 for demonstration purposes
	print(f"{len(audio_array)} amplitude samples when sampled at 44.1kHz") 
	# x samples, 1632384 samples when using demo audio "res/audio/voice-message-1.mp3"

	audio_array = load_audio("res/audio/voice-message-1.mp3", sr=16000) # Whisper internally operates at 16kHz
	print(f"{len(audio_array)} amplitude samples when sampled at 16kHz") 
	#y samples, 592248 samples when using demo audio "res/audio/voice-message-1.mp3"
	# y samples = x samples/2.75625 (because 44100/16000=2.75625)

	audio_array = load_audio("res/audio/voice-message-1.mp3") # if no sampling rate is provided, Whisper will use its default sampling rate of 16000/16kHz.
	print(f"{len(audio_array)} amplitude samples when sampled at default Whisper sample rate") 
	# z samples, same number of samples as y
	#592248 samples

def demo_pad_array():
	audio_array = load_audio("res/audio/voice-message-1.mp3") # if no sampling rate is provided, Whisper will load audio using its default sampling rate of 16000/16kHz.
	print(f"{len(audio_array)} amplitude samples when sampled at default Whisper sample rate") 
	#592248 samples

	audio_array = pad_or_trim(audio_array)
	print(f"{len(audio_array)} samples after trimming") # audio is longer than 30s/480000 samples, will be trimmed to 480000 samples, 16kHz sample rate * 30 second chunk size

	audio_array = load_audio("res/audio/voice-message-4.ogg") # if no sampling rate is provided, Whisper will use its default sampling rate of 16000/16kHz.
	print(f"{len(audio_array)} amplitude samples when sampled at default Whisper sample rate") 
	#158400 samples

	audio_array = pad_or_trim(audio_array)
	print(f"{len(audio_array)} samples after padding") # audio is shorter than 30s/480000 samples, will be padded or trimmed to 480000 samples, 16kHz sample rate * 30 second chunk size

def demo_compute_metrics():
	"Demos the word error rate (WER) calculation provided by Evaluate using the manually verified transcript for voice-message-1-split-1.mp3 as reference, compared to the predicted text output by default Whisper tiny.en model. For more information on WER in speech recognition, please see https://huggingface.co/learn/audio-course/en/chapter5/evaluation"
	metric = load("wer")
	reference = " Yeah, I totally get that. I absolutely love the energy of the storms. Like, yeah, of course, you know, there's damage or danger, like that's no good. But the energy? It's just such cool shit. We get these just like wild thunderstorms here and so often they'll just like rip through in like 15 to 30 minutes." # 57 words
	prediction = " Yeah, I totally get that. I absolutely love the energy. The storm is like, yeah, of course, you know, there's damage or danger, like that's no good. But with the energy, it's just such cool shit. We get these just like wild thunderstorms here. And so often they'll just like rips through in like 15 to 30 minutes." 
		# substitutions: 8 ('here. And' counted as one substitution)
		# insertions: 2
		# deletions: 1
		# expected WER: (8+2+1)/57~=0.192982

	wer = metric.compute(references=[reference], predictions=[prediction])

	print(wer) # 0.19298245614035087

	if wer != None:
		wer = wer * 100 # type: ignore
		
	print(wer)

	return {'wer': wer}

# demo_load_audio_output()
# demo_pad_array()
demo_compute_metrics()