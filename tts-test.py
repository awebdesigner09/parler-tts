import torch
import torch.backends.cuda as cuda_backends
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread
import os

# Check if CUDA is available to PyTorch
print(f"Is CUDA available? {torch.cuda.is_available()}")

# If CUDA is available, check the detected version
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    # Try accessing cudnn directly (might still fail if cuDNN itself is the issue)
    try:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Is cuDNN available? {torch.backends.cudnn.is_available()}")
        print(f"Is cuDNN enabled? {torch.backends.cudnn.enabled}")

         # Performance optimizations for CUDA - Use full path and check availability
        print("Applying CUDA performance optimizations...")
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.enabled = True # This is often True by default if available
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        # The attribute below might vary depending on PyTorch version, ensure it exists
        if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
             torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
    except AttributeError as e:
        print(f"Could not access cuDNN attributes: {e}")
else:
    print("CUDA not available. PyTorch might be CPU-only or CUDA drivers/toolkit are missing/misconfigured.")

# Check the exact PyTorch build information
print(f"PyTorch version: {torch.__version__}")

# Performance optimizations for CUDA
# cuda_backends.cudnn.benchmark = True
# cuda_backends.cudnn.enabled = True
# cuda_backends.cudnn.deterministic = False
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Force CUDA device selection
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Usually not needed if PyTorch detects the GPU
torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "parler-tts/parler-tts-mini-v1"

print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# need to set padding max length
max_length = 50

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16
    # use_cache=True,
    # device_map="cuda:0",
    # low_cpu_mem_usage=True,
).to(torch_device).eval()

# Enable torch.compile for better performance
# if hasattr(torch, 'compile') and torch.cuda.is_available():
#     model = torch.compile(model, mode="reduce-overhead")

# Warmup run (simplified)
inputs = tokenizer("Warmup text", return_tensors="pt", padding=True).to(torch_device)
model_kwargs = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask,
    "prompt_input_ids": inputs.input_ids,
    "prompt_attention_mask": inputs.attention_mask,
}
_ = model.generate(**model_kwargs)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

def generate(text, description, play_steps_in_s=0.5):
    with torch.cuda.amp.autocast(enabled=True):
        play_steps = int(frame_rate * play_steps_in_s)
        streamer = ParlerTTSStreamer(
            model, 
            device=torch_device, 
            play_steps=play_steps
        )
        # tokenization with optimized settings
        inputs = tokenizer(
            description, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=max_length,
            truncation=True
        ).to(torch_device)
        
        prompt = tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=max_length,
            truncation=True
        ).to(torch_device)

        # Optimized generation settings
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "prompt_input_ids": prompt.input_ids,
            "attention_mask": inputs.attention_mask,
            "prompt_attention_mask": prompt.attention_mask,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.7,
            "min_new_tokens": 10,
            "use_cache": True,
            "num_beams": 1,
            "length_penalty": 1.0,
            # "early_stopping": True,
            "max_new_tokens": None,
            # "do_prefill": True,
            "typical_p": 0.95,
            "pad_token_id": model.generation_config.pad_token_id,
            "repetition_penalty": 1.0,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
            print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 4)} seconds")
            yield sampling_rate, new_audio

def process_text(text, description, output_filename, chunk_size_in_s=0.5):
    """Process a single text input and save it to a WAV file.
    
    Args:
        text (str): The text to convert to speech
        description (str): The description of how to speak the text
        output_filename (str): The filename to save the audio to
        chunk_size_in_s (float): The size of each audio chunk in seconds
    """
    all_chunks = []
    
    print(f"\nProcessing: {text}")
    print(f"Description: {description}")
    
    for (sampling_rate, audio_chunk) in generate(text, description, chunk_size_in_s):
        if not isinstance(audio_chunk, torch.Tensor):
            audio_chunk = torch.from_numpy(audio_chunk)
        all_chunks.append(audio_chunk)
        print(f"Generated chunk of shape: {audio_chunk.shape}")
    
    final_audio = torch.cat(all_chunks, dim=0)
    import soundfile as sf
    sf.write(output_filename, final_audio.cpu().numpy(), sampling_rate)
    print(f"Saved audio to: {output_filename}")

def process_batch(test_cases, batch_size=2):
    """Process multiple test cases in batches for parallel processing"""
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i + batch_size]
        threads = []
        for case in batch:
            thread = Thread(target=process_text, kwargs={
                'text': case["text"],
                'description': case["description"],
                'output_filename': case["filename"],
                'chunk_size_in_s': 0.5
            })
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

# Test multiple texts with different lengths and descriptions
test_cases = [
    {
        "text": "This is a short test.",
        "description": "Speaking calmly and clearly.",
        "filename": "short_test.wav"
    },
    {
        "text": "This is a medium length test of the text to speech system. It contains multiple sentences to process.",
        "description": "Speaking at a moderate pace with good articulation.",
        "filename": "medium_test.wav"
    },
    {
        "text": "This is a longer test with multiple sentences. We want to see how the system handles longer pieces of text. This will help us understand the capabilities and limitations of the model. Let's see how it performs.",
        "description": "Speaking professionally with clear pronunciation.",
        "filename": "long_test.wav"
    }
]

# Process each test case in batches
process_batch(test_cases, batch_size=2)