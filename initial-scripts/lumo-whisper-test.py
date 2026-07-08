import json
import os
import torch
from typing import Dict, List, Optional
from pathlib import Path
import whisper
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

class WhisperBatchTranscriber:
    """
    Batch transcription system using multiple Whisper models
    """
    
    def __init__(self, models: List[str] = None, device: str = None):
        """
        Initialize the transcriber with specified Whisper models
        
        Args:
            models: List of Whisper model names (e.g., ['tiny', 'base', 'small', 'medium', 'large'])
            device: Device to run on ('cpu' or 'cuda'). Auto-detects if None.
        """
        self.models = models or ['tiny', 'base', 'small', 'medium', 'large']
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all models at initialization
        print(f"Loading {len(self.models)} Whisper models on {self.device}...")
        self.loaded_models = {}
        for model_name in self.models:
            try:
                self.loaded_models[model_name] = whisper.load_model(model_name).to(self.device)
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio file to match Whisper requirements
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (Whisper expects 16kHz)
            
        Returns:
            Preprocessed audio as numpy array
        """
        # Load audio with librosa (handles various formats)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio
    
    def transcribe_single_file(self, 
                              audio_path: str, 
                              model_name: str,
                              language: Optional[str] = None) -> Dict:
        """
        Transcribe a single audio file with a specific model
        
        Args:
            audio_path: Path to audio file
            model_name: Name of Whisper model to use
            language: Language code (optional, auto-detected if None)
            
        Returns:
            Dictionary with transcription results
        """
        try:
            model = self.loaded_models[model_name]
            
            # Preprocess audio
            audio = whisper.load_audio(audio_path)
            
            # Handle long audio by splitting into chunks if necessary
            # Whisper processes up to 30 seconds at a time
            chunk_duration = 30 * 16000  # 30 seconds in samples
            if len(audio) > chunk_duration:
                # Split into chunks
                chunks = [audio[i:i+chunk_duration] for i in range(0, len(audio), chunk_duration)]
                
                # Transcribe each chunk
                transcripts = []
                for chunk in chunks:
                    result = model.transcribe(chunk, language=language)
                    transcripts.append(result['text'])
                
                full_text = " ".join(transcripts)
            else:
                # Single chunk
                result = model.transcribe(audio, language=language)
                full_text = result['text']
            
            return {
                'file_path': audio_path,
                'model': model_name,
                'transcript': full_text,
                'status': 'success',
                'error': None
            }
            
        except Exception as e:
            return {
                'file_path': audio_path,
                'model': model_name,
                'transcript': None,
                'status': 'error',
                'error': str(e)
            }
    
    def process_batch(self, 
                     audio_files: List[str], 
                     output_dir: str = './res/transcriptions',
                     num_workers: int = 4) -> Dict:
        """
        Process a batch of audio files with all models
        
        Args:
            audio_files: List of paths to audio files
            output_dir: Directory to save JSON output
            num_workers: Number of parallel processes
            
        Returns:
            Dictionary containing all transcription results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        total_files = len(audio_files) * len(self.models)
        processed = 0
        
        print(f"Starting batch transcription: {total_files} tasks ({len(audio_files)} files × {len(self.models)} models)")
        
        # Use multiprocessing for CPU-bound transcription tasks
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {}
            for audio_file in audio_files:
                if not os.path.exists(audio_file):
                    print(f"Warning: File not found - {audio_file}")
                    continue
                    
                for model_name in self.models:
                    future = executor.submit(
                        self.transcribe_single_file, 
                        audio_file, 
                        model_name
                    )
                    futures[future] = (audio_file, model_name)
            
            # Collect results as they complete
            for future in as_completed(futures):
                audio_file, model_name = futures[future]
                try:
                    result = future.result()
                    
                    # Organize results by file
                    if audio_file not in results:
                        results[audio_file] = {'file_path': audio_file, 'models': {}}
                    
                    results[audio_file]['models'][model_name] = result
                    
                    processed += 1
                    progress = (processed / total_files) * 100
                    print(f"Progress: {processed}/{total_files} ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"Error processing {audio_file} with {model_name}: {e}")
        
        # Save results to JSON
        output_file = os.path.join(output_dir, 'whisper_batch_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Transcription complete! Results saved to: {output_file}")
        return results

def main():
    """
    Example usage of the batch transcriber
    """
    # Configuration
    AUDIO_DIR = './res/audio'  # Directory containing your audio files
    OUTPUT_DIR = './res/transcriptions'
    
    # Models to evaluate (you can customize this list)
    MODELS = ['tiny', 'base', 'small']  # Start with smaller models for testing
    
    # Get list of audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(Path(AUDIO_DIR).glob(f'*{ext}')))
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Initialize transcriber
    transcriber = WhisperBatchTranscriber(models=MODELS)
    
    # Process batch
    results = transcriber.process_batch(
        audio_files=[str(f) for f in audio_files],
        output_dir=OUTPUT_DIR,
        num_workers=min(len(MODELS), 4)  # Adjust workers based on available RAM
    )
    
    # Print summary
    print("\n=== TRANSCRIPTION SUMMARY ===")
    for file_path, file_results in results.items():
        print(f"\nFile: {os.path.basename(file_path)}")
        for model_name, result in file_results['models'].items():
            status = "✓" if result['status'] == 'success' else "✗"
            print(f"  {model_name:8s} {status}")
            if result['status'] == 'success':
                preview = result['transcript'][:100] + "..." if len(result['transcript']) > 100 else result['transcript']
                print(f"           '{preview}'")
            else:
                print(f"           Error: {result['error']}")

if __name__ == "__main__":
    main()