class DataEntry:
    """
    Represents a structured data entry containing audio transcription metadata and content.
    
    This class encapsulates information about a speech/audio file, including the source 
    filename, timing information, the model used for transcription, verification status, 
    the full transcript text, and segmented portions of the transcript. The getter and setter methods and documentation for this class was generated using Lumo AI based on json metadata fields provided to it.
    
    Attributes:
        file_name (str): The name/path of the source audio file.
        speech_ends_at (float): Timestamp in seconds indicating where speech ends in the audio.
        model_name (str): Name of the AI/transcription model used to generate the transcript.
        manually_verified (bool): Whether the transcript has been reviewed by a human.
        transcript (str): The complete text transcription of the audio content.
        segments (list): List of segmented portions of the transcript (e.g., by speaker, timestamp, or topic).
    
    Example:
        >>> entry = DataEntry(
        ...     file_name="interview.wav",
        ...     speech_ends_at=180.5,
        ...     model_name="whisper-v3",
        ...     manually_verified=True,
        ...     transcript="Welcome to our podcast...",
        ...     segments=[{"start": 0, "end": 10, "text": "Welcome..."}]
        ... )
        >>> print(entry.file_name)
        interview.wav
    
    Note:
        All properties are accessible via getter/setter methods with type validation:
        - file_name and transcript must be strings
        - speech_ends_at must be a non-negative numeric value
        - manually_verified must be a boolean
        - segments must be a list
    """
    
    def __init__(self, file_name: str, speech_ends_at: float, model_name: str, 
                 manually_verified: bool, transcript: str, segments: list) -> None:
        self._file_name = file_name
        self._speech_ends_at = speech_ends_at
        self._model_name = model_name
        self._manually_verified = manually_verified
        self._transcript = transcript
        self._segments:list[Segment] = segments

    @property
    def file_name(self) -> str:
        """Get the audio file name."""
        return self._file_name

    @file_name.setter
    def file_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("file_name must be a string")
        self._file_name = value

    @property
    def speech_ends_at(self) -> float:
        """Get the timestamp when speech ends."""
        return self._speech_ends_at

    @speech_ends_at.setter
    def speech_ends_at(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("speech_ends_at must be a numeric value")
        if value < 0:
            raise ValueError("speech_ends_at cannot be negative")
        self._speech_ends_at = float(value)

    @property
    def model_name(self) -> str:
        """Get the transcription model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("model_name must be a string")
        self._model_name = value

    @property
    def manually_verified(self) -> bool:
        """Get the manual verification status."""
        return self._manually_verified

    @manually_verified.setter
    def manually_verified(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("manually_verified must be a boolean")
        self._manually_verified = value

    @property
    def transcript(self) -> str:
        """Get the full transcript text."""
        return self._transcript

    @transcript.setter
    def transcript(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("transcript must be a string")
        self._transcript = value

    @property
    def segments(self) -> list:
        """Get the list of transcript segments."""
        return self._segments

    @segments.setter
    def segments(self, value: list) -> None:
        if not isinstance(value, list):
            raise TypeError("segments must be a list")
        self._segments = value
        
    def set_segments_from_objects(self, segments:list[Segment]):
        # TODO uncertain if I will need this; may come back to it
        pass

    def convert_segments_to_objects(self, segments:list) -> list[Segment]:
        objects_list:list[Segment] = []
        for i in segments:
            segment: Segment = Segment(id=i['id'], start=i['start'], end=i['end'], text=i['text'], words=i['words'])
            objects_list.append(segment)
        return objects_list

    def entry_to_list(self) -> list:
        segments_list:list = []
        for segment in self._segments:
            segments_list.append(segment.segment_to_list())

        output_list:list = [{"file_name": self._file_name}, {"speech_ends_at":self._speech_ends_at}, {"model_name": self._model_name}, {"manually_verified":self._manually_verified}, {"transcript": self._transcript}, {"segments": segments_list}]
        
        return output_list
    
    def entry_to_dict(self) -> dict:
        segments_list:list = []
        for segment in self._segments:
            if isinstance(segment, Segment):
                segments_list.append(segment.segment_to_list())
            else:
                segments_list.append(segment)
            
        output_dict:dict = {"file_name": self._file_name, "speech_ends_at":self._speech_ends_at, "model_name": self._model_name, "manually_verified":self._manually_verified, "transcript": self._transcript, "segments": segments_list}
        
        print(output_dict["file_name"])
        
        return output_dict


class Segment:
    """
    Represents a single segment of transcribed speech with temporal metadata.
    
    Each segment contains a portion of the transcript with associated timing 
    information, identifying ID, and optional word-level breakdown as a list. The getter and setter methods and documentation for this class was generated using Lumo AI based on json metadata fields provided to it.
    
    Attributes:
        id (int): Unique identifier for this segment.
        start (float): Start timestamp in seconds relative to audio beginning.
        end (float): End timestamp in seconds relative to audio beginning.
        text (str): Transcribed text for this segment.
        words (list): List of word-level breakdown entries with timing/spoken data.
    
    Example:
        >>> seg = Segment(
        ...     id=0,
        ...     start=1.38,
        ...     end=3.1,
        ...     text="Yeah, I totally get that.",
        ...     words=[]
        ... )
        >>> print(seg.duration)
        1.72
    
    Note:
        - end must be greater than or equal to start
        - id should be non-negative
        - All timestamps are in seconds
        - words is a list of word objects (typically dicts with 'word', 'start', 'end' keys)
    """
    
    def __init__(self, id: int, start: float, end: float, text: str, words: list) -> None:
        self._id = id
        self._start = start
        self._end = end
        self._text = text
        self._words = words if words is not None else []
    
    # Property: id
    @property
    def id(self) -> int:
        """Get the unique segment identifier."""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("id must be an integer")
        if value < 0:
            raise ValueError("id cannot be negative")
        self._id = value

    # Property: start
    @property
    def start(self) -> float:
        """Get the start timestamp."""
        return self._start

    @start.setter
    def start(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("start must be a numeric value")
        if value < 0:
            raise ValueError("start cannot be negative")
        self._start = float(value)

    # Property: end
    @property
    def end(self) -> float:
        """Get the end timestamp."""
        return self._end

    @end.setter
    def end(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("end must be a numeric value")
        if value < 0:
            raise ValueError("end cannot be negative")
        if value < self._start:
            raise ValueError("end cannot be less than start")
        self._end = float(value)

    # Property: text
    @property
    def text(self) -> str:
        """Get the transcribed text."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("text must be a string")
        self._text = value

    # Property: words (changed from dict to list)
    @property
    def words(self) -> list:
        """Get the word-level breakdown list."""
        return self._words

    @words.setter
    def words(self, value: list) -> None:
        if not isinstance(value, list):
            raise TypeError("words must be a list")
        self._words = value

    # Computed property: duration
    @property
    def duration(self) -> float:
        """Calculate the segment duration in seconds."""
        return self._end - self._start

    # Method: Contains another segment temporally
    def overlaps_with(self, other_segment: 'Segment') -> bool:
        """
        Check if this segment overlaps with another segment.
        
        Args:
            other_segment: Another Segment instance to compare against.
            
        Returns:
            bool: True if there's temporal overlap, False otherwise.
        """
        return not (self._end <= other_segment._start or self._start >= other_segment._end)

    # String representation
    def __repr__(self) -> str:
        return f"Segment(id={self._id}, start={self._start:.2f}s, end={self._end:.2f}s, text='{self._text[:30]}...')"
    
    def __str__(self) -> str:
        return f"[{self._id}] {self._text.strip()} ({self._start:.2f}-{self._end:.2f}s)"
    
    def segment_to_list(self) -> list:
        output_list:list = [{"id": self._id}, {"start":self._start}, {"end": self._end}, {"text":self._text}, {"words": self._words}]
        
        return output_list
    
    def segment_to_dict(self) -> dict:
        output_dict:dict = {"id": self._id, "start":self._start, "end": self._end, "text":self._text, "words": self._words}
        
        return output_dict