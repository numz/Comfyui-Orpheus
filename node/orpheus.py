import os
import time
import logging

import wave
import folder_paths
import hashlib
import torchaudio
from .decoder import convert_to_audio as orpheus_convert_to_audio

from llama_cpp import Llama


def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])


BOOLEAN = ("BOOLEAN", {"default": True})
STRING = ("STRING", {"default": ""})

# Model parameters
MAX_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe", "pierre", "amelie", "marie","jana", "thomas", "max", "유나", "준서", "长乐", "白芷" "javi", "sergio", "maria", "pietro", "giulia", "carlo"]
DEFAULT_VOICE = "pierre"  # Best voice according to documentation

CUSTOM_TOKEN_PREFIX = "<custom_token_"


def format_prompt(prompt, voice=DEFAULT_VOICE):
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE 
        
    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers for the LM Studio API
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config
    
    return f"{special_start}{formatted_prompt}{special_end}"

def turn_token_into_id(token_string, index):
    """Convert token string to numeric ID for audio processing."""
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        #print(f"Last token: {last_token}")
        try:
            number_str = last_token[14:-1]
            # print(f"Number string: {number_str}")
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            # print(f"Token ID: {token_id}")
            return token_id
        except ValueError:
            return None
    else:
        return None

def convert_to_audio(multiframe, count):
    """Convert token frames to audio."""
    # Import here to avoid circular imports
   
    return orpheus_convert_to_audio(multiframe, count)

def my_tokens_decoder(token_gen):
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer = []
    count = 0
    audio_samples_buffer = []
    #print(token_gen)
    #tgen = extract_custom_tokens(token_gen)
    for token_text in token_gen:
        #print("Token text:", token_text)
        token = turn_token_into_id(token_text, count)
        #print("Token ID:", token)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    audio_samples_buffer.append(audio_samples)
    return audio_samples_buffer

def extract_custom_tokens(token_string):
    """
    Extract all custom tokens from a string and return them as an array.
    Example: "<custom_token_465456><custom_token_7867>" -> ["<custom_token_465456>", "<custom_token_7867>"]
    """
    #print(token_string)
    tokens = []
    i = 0
    while i < len(token_string):
        # Find the start of a custom token
        start_pos = token_string.find(CUSTOM_TOKEN_PREFIX, i)
        if start_pos == -1:
            break
            
        # Find the end of this token
        end_pos = token_string.find(">", start_pos)
        if end_pos == -1:
            break
            
        # Extract the complete token
        token = token_string[start_pos:end_pos+1]
        tokens.append(token)
        
        # Move past this token
        i = end_pos + 1
    
    return tokens

def tokens_decoder_sync(syn_token_gen, output_file=None):
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    def my_producer(syn_token_gen):
        return my_tokens_decoder(extract_custom_tokens(syn_token_gen))
    
    tokens = my_producer(syn_token_gen)

    for audio_chunk in tokens:
        wav_file.writeframes(audio_chunk)

    if wav_file:
        wav_file.close()


def load_model(model_path):
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # Context length to use
        n_threads=12,            # Number of CPU threads to use
        n_gpu_layers=32,
        verbose=False
    )
    return llm


class orpheus:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
      unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
      #print(unet_names)
      return {
        "required": {
          "model_name": (unet_names, ),
          "voice": (AVAILABLE_VOICES,),
          "prompt": ("STRING", {"default": "Hello, I am Orpheus, an AI assistant with emotional speech capabilities.","multiline": True})
        },
      }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "Orpheus ⛓️"

    @classmethod
    def IS_CHANGED(s, model_name, voice,prompt, **kwargs):
        m = hashlib.sha256()
        m.update(model_name.encode() + voice.encode() + prompt.encode())
        return m.digest().hex()

    def execute(self, model_name, voice, prompt, **kwargs):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(folder_paths.get_output_directory(), f"orpheus_{voice}_{timestamp}.wav")
        model_path = folder_paths.get_full_path("unet_gguf", model_name)
        #print(model_path)
        llm = load_model(model_path)
        prompt = format_prompt(prompt, voice=voice)
        generation_kwargs = {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repeat_penalty": REPETITION_PENALTY
        }

        res = llm(prompt, **generation_kwargs) # Res is a dictionary
        tokens_decoder_sync(res['choices'][0]['text'], output_file)

        # open file as a tensor
        waveform, sample_rate = torchaudio.load(output_file)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )


class orpheusAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
      unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
      #print(unet_names)
      return {
        "required": {
          "model_name": (unet_names, ),
          "voice": (AVAILABLE_VOICES,),
          "prompt": ("STRING", {"default": "Hello, I am Orpheus, an AI assistant with emotional speech capabilities.","multiline": True}),
          "max_tokens" :("INT", {"default": 8192, "min": 4096, "max": 131072, "step": 1}),
          "temperature" :("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
          "top_p" :("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
          "repeat_penalty" :("FLOAT", {"default": 1.1, "min": 0.0, "max": 1.5, "step": 0.01}),
        },
      }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "Orpheus ⛓️"

    @classmethod
    def IS_CHANGED(s, model_name, voice,prompt, max_tokens, temperature, top_p, repeat_penalty, **kwargs):
        m = hashlib.sha256()
        m.update(model_name.encode() + voice.encode() + prompt.encode() + str(max_tokens).encode() + str(temperature).encode() + str(top_p).encode() + str(repeat_penalty).encode())
        return m.digest().hex()

    def execute(self, model_name, voice, prompt, max_tokens, temperature, top_p, repeat_penalty, **kwargs):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(folder_paths.get_output_directory(), f"orpheus_{voice}_{timestamp}.wav")
        model_path = folder_paths.get_full_path("unet_gguf", model_name)
        #print(model_path)
        llm = load_model(model_path)
        prompt = format_prompt(prompt, voice=voice)
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty
        }

        res = llm(prompt, **generation_kwargs) # Res is a dictionary
        tokens_decoder_sync(res['choices'][0]['text'], output_file)

        # open file as a tensor
        waveform, sample_rate = torchaudio.load(output_file)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

NODE_CLASS_MAPPINGS = {
    "orpheus": orpheus,
    "orpheusAdvanced": orpheusAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "orpheus": "Orpheus",
    "orpheusAdvanced": "Orpheus Advanced",
}
