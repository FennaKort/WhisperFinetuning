import json
import os
from typing import List, Optional, Dict
from pathlib import Path
import torch
import whisper
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# CRITICAL FIX: Define worker_task at MODULE LEVEL (not nested inside class)
# This allows multiprocessing to serialize/pickle the function
# =============================================================================

def _worker_task(args: tuple, device: str = "cpu") -> dict:
    """
    Worker function for batch transcription (module-level for pickling).
    
    Args:
        args: Tuple of (file_path, model_name)
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Dictionary with transcription result
    """
    file_path, model_name = args
    
    try:
        # Load model fresh in worker process
        model = whisper.load_model(model_name).to(device)
        if device == "cuda":
            model = model.half()
        
        # Load audio (handles resampling, format conversion)
        audio_tensor = whisper.load_audio(file_path).to(device)
        
        # Handle long audio (>30 seconds / 480,000 samples at 16kHz)
        SAMPLE_RATE = 16000
        MAX_SAMPLES = SAMPLE_RATE * 30
        
        if len(audio_tensor) > MAX_SAMPLES:
            chunks = [audio_tensor[i:i+MAX_SAMPLES] 
                     for i in range(0, len(audio_tensor), MAX_SAMPLES)]
            texts = [model.transcribe(c)['text'] for c in chunks]
            text = " ".join(texts)
        else:
            text = model.transcribe(audio_tensor)['text']
        
        return {
            'file_path': os.path.abspath(file_path),
            'model': model_name,
            'transcript': text,
            'status': 'success',
            'error': None
        }
    except FileNotFoundError:
        return {
            'file_path': file_path,
            'model': model_name,
            'transcript': None,
            'status': 'error',
            'error': 'File not found'
        }
    except Exception as e:
        return {
            'file_path': file_path,
            'model': model_name,
            'transcript': None,
            'status': 'error',
            'error': str(e)
        }


class WhisperBatchTranscriber:
    """
    Batch transcription system using multiple Whisper models.
    No scipy/librosa dependencies - uses Whisper's native audio loading.
    """
    
    def __init__(self, models: List[str] = None, device: str = None):
        self.models = models or ['tiny', 'base', 'small', 'medium', 'large']
        
        # Auto-detect device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"Initializing {len(self.models)} Whisper models on '{self.device}'...")
        self.loaded_models = {}
        
        # Pre-load models in main process
        for model_name in self.models:
            try:
                model = whisper.load_model(model_name).to(self.device)
                if self.device == "cuda":
                    model = model.half()
                self.loaded_models[model_name] = model
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
    
    def transcribe_single_file(self, 
                              audio_path: str, 
                              model_name: str,
                              language: Optional[str] = None) -> Dict:
        """
        Transcribe using pre-loaded models in main process (non-parallel mode).
        """
        try:
            model = self.loaded_models[model_name]
            audio_tensor = whisper.load_audio(audio_path).to(self.device)
            
            SAMPLE_RATE = 16000
            MAX_SAMPLES = SAMPLE_RATE * 30
            
            if len(audio_tensor) > MAX_SAMPLES:
                chunks = [audio_tensor[i:i+MAX_SAMPLES] 
                         for i in range(0, len(audio_tensor), MAX_SAMPLES)]
                texts = [model.transcribe(c, language=language)['text'] for c in chunks]
                text = " ".join(texts)
            else:
                text = model.transcribe(audio_tensor, language=language)['text']
            
            return {
                'file_path': os.path.abspath(audio_path),
                'model': model_name,
                'transcript': text,
                'status': 'success',
                'error': None,
                'duration_estimate': len(audio_tensor) / SAMPLE_RATE
            }
        except Exception as e:
            return {
                'file_path': audio_path,
                'model': model_name,
                'transcript': None,
                'status': 'error',
                'error': str(e)
            }
    
    def process_batch_parallel(self,
                               audio_files: List[str],
                               output_dir: str = './res/transcriptions',
                               num_workers: int = 2) -> Dict:
        """
        Parallel batch processing (fixes the pickle error).
        Each worker loads its own model copies.
        
        WARNING: Memory usage scales with num_workers × number_of_models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        valid_files = [f for f in audio_files if os.path.isfile(f)]
        
        for f in audio_files:
            if not os.path.isfile(f):
                print(f"Warning: Skipping missing file: {f}")
        
        total_tasks = len(valid_files) * len(self.models)
        print(f"Starting parallel batch: {len(valid_files)} files × {len(self.models)} models = {total_tasks} tasks")
        
        # Build task list
        tasks = [(f, m) for f in valid_files for m in self.models]
        
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_worker_task, task, self.device): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                file_path, model_name = futures[future]
                try:
                    result = future.result()
                    
                    if file_path not in results:
                        results[file_path] = {'file_path': file_path, 'models': {}}
                    results[file_path]['models'][model_name] = result
                    
                    processed_count += 1
                    if processed_count % 5 == 0 or processed_count == total_tasks:
                        pct = 100 * processed_count / total_tasks
                        print(f"Progress: {processed_count}/{total_tasks} ({pct:.1f}%)")
                        
                except Exception as e:
                    print(f"Worker error for {file_path}/{model_name}: {e}")
        
        # Save results
        output_file = os.path.join(output_dir, 'batch_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDone! Results saved to {output_file}")
        return results
    
    def process_batch_sequential(self,
                                 audio_files: List[str],
                                 output_dir: str = './res/transcriptions') -> Dict:
        """
        Sequential batch processing (no parallelism, memory-efficient).
        Uses pre-loaded models from main process.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        total_tasks = len(audio_files) * len(self.models)
        processed = 0
        
        print(f"Starting sequential batch: {total_tasks} tasks")
        
        for audio_file in audio_files:
            if not os.path.isfile(audio_file):
                print(f"Skipping missing file: {audio_file}")
                continue
                
            results[audio_file] = {'file_path': audio_file, 'models': {}}
            
            for model_name in self.models:
                result = self.transcribe_single_file(audio_file, model_name)
                results[audio_file]['models'][model_name] = result
                
                processed += 1
                if processed % 5 == 0 or processed == total_tasks:
                    pct = 100 * processed / total_tasks
                    print(f"Progress: {processed}/{total_tasks} ({pct:.1f}%)")
        
        output_file = os.path.join(output_dir, 'batch_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDone! Results saved to {output_file}")
        return results


if __name__ == "__main__":
    AUDIO_DIR = "./res/audio"
    FILES = list(Path(AUDIO_DIR).glob("*.wav")) + list(Path(AUDIO_DIR).glob("*.mp3")) + list(Path(AUDIO_DIR).glob("*.ogg"))
    MODELS_TO_TEST = ["tiny", "base"]
    
    if not FILES:
        print("No audio files found.")
    else:
        transcriber = WhisperBatchTranscriber(models=MODELS_TO_TEST)
        
        # Option 1: Parallel (faster, more RAM)
        # transcriber.process_batch_parallel([str(f) for f in FILES], num_workers=2)
        
        # Option 2: Sequential (slower, less RAM - safer for large models)
        transcriber.process_batch_sequential([str(f) for f in FILES])