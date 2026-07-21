import json
import numpy as np
from whisper import load_audio

class NumpyTypeEncoder(json.JSONEncoder):
    """JSON serialization method for NumPy datatypes by Jonathon Lurie. Used to encode audio amplitude arrays for output in JSON files. From https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65"""
    def default(self, o):
        if isinstance(o, np.generic):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
    
