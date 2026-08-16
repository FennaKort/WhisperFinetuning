from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper import load_audio
import torch
import os
from evaluate import load

class FineTunedTranscriber:
    def __init__(self):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

    def transcribe(self, audio_file:str, processor:WhisperProcessor, model:WhisperForConditionalGeneration):
        # Load the audio file
        sr=16000
        audio = load_audio(audio_file, sr=sr)  

        # Prepare the audio input for the model
        input_features = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)

        language_token_id = processor.tokenizer.convert_tokens_to_ids('<|en|>')
        forced_decoder_ids = [[1, language_token_id]]

        # Generate transcription with forced_decoder_ids for English
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        # Decode the transcription
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription
    
    def batch_transcribe(self, audio_list:list, processor:WhisperProcessor, model:WhisperForConditionalGeneration):
        transcription_list:list = []
        for audio_file in audio_list:
            transcription_list.append(self.transcribe(audio_file,processor,model))
        return transcription_list

    def compute_metrics(self, verified_transcriptions_list:list, predicted_transcriptions_list:list):
        "word error rate (WER) calculation provided by Evaluate using manually verified transcripts as reference, compared to the predicted text output by a Whisper model. For more information on WER in speech recognition, please see https://huggingface.co/learn/audio-course/en/chapter5/evaluation"
        metric = load("wer")
        wer = 0.0
        print(len(verified_transcriptions_list))
        if (len(verified_transcriptions_list)==len(predicted_transcriptions_list)): #has same number of transcriptions
            i = 0
            while i < len(verified_transcriptions_list):
                predictions = predicted_transcriptions_list[i]
                references = verified_transcriptions_list[i]
                i += 1
                current_wer = metric.compute(predictions=predictions, references=references)
                print(f'current wer: {current_wer}')
                wer += current_wer
                print(wer)
            
            if wer != None:
                wer / len(verified_transcriptions_list)
                print(wer)
                wer = wer * 100 # type: ignore  
                
        
        # wer = metric.compute(predictions=[" This is just a store test transcription that's time to test the 19 model and it is outside of the dataset entirely.  This is another short transcription that I'm using to test the fine-tuned model and see how it's performing on a similar set of words from another session."], references=[" This is just a short test transcription that I'm using to test the fine tuned model and it is outside of the data set entirely. And this is another short transcription that I'm using to test the fine tuned model and see how it's performing on a similar set of words from another session."])
                
        # if wer != None:
        #     wer / len(verified_transcriptions_list)
        #     wer = wer * 100 # type: ignore  
             
        return {'wer': wer}

    def test_finetuned_model(self, audio_list:list, verified_transcriptions_list:list, model_dir:str, wer:float):
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model.to(self.device)
        processor = WhisperProcessor.from_pretrained(model_dir)

        print(f"Transcribing with {model_dir} test wer {wer}%:")
        transcriptions = self.batch_transcribe(audio_list, processor, model)
        print(transcriptions)
        wer = self.compute_metrics(verified_transcriptions_list,transcriptions)
        print(f"{wer}\n")

def main():
    audio_list: list = ["res/post-training-test/post-training-test-01.mp3","res/post-training-test/post-training-test-02.mp3"]
    verified_transcriptions_list: list = [[" This is just a short test transcription that I'm using to test the fine tuned model and it is outside of the data set entirely."], [" And this is another short transcription that I'm using to test the fine tuned model and see how it's performing on a similar set of words from another session."]]
    transcriber = FineTunedTranscriber()

    # transcription with HF whisper tiny.en
    transcriber.test_finetuned_model(audio_list,verified_transcriptions_list, "openai/whisper-tiny.en", 18.23)
    # sub: 4
    # del: 3
    # words: 25

    # transcription with the finetuned model and processor
    transcriber.test_finetuned_model(audio_list,verified_transcriptions_list,"./fine-tuned-model/tiny.en-2026-08-11", 00.00)

if __name__ == "__main__":
	main()
