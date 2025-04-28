from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from enum import Enum
from pydantic import BaseModel
import torch
# import torch.backends.cuda as cuda_backends
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
import io
import soundfile as sf
from typing import List
import pydub

# Performance optimizations for CUDA (wrapped in checks)
if torch.cuda.is_available():
    print("CUDA available. Applying performance optimizations...")
    try:
        # Check if cudnn is accessible before trying to set attributes
        if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True # Often true by default if available
            print("cuDNN optimizations applied.")
        else:
            print("cuDNN not available or accessible via torch.backends.cudnn.")

        # Set matmul precision attributes (these are usually safe even if cudnn isn't fully utilized)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        print("CUDA matmul optimizations applied.")

    except AttributeError as e:
        print(f"Could not apply some CUDA optimizations: {e}")
else:
    print("CUDA not available. Skipping CUDA-specific performance optimizations.")

# Set up constants
TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16  # Using float16 for better performance
MODEL_NAME = "parler-tts/parler-tts-mini-v1"
# BATCH_SIZE = 1  # Adjust based on your GPU memory

# Define available speakers (top 20 from mini model)
class SpeakerId(str, Enum):
    WILL = "Will"
    ERIC = "Eric"
    LAURA = "Laura"
    ALISA = "Alisa"
    PATRICK = "Patrick"
    ROSE = "Rose"
    JERRY = "Jerry"
    JORDAN = "Jordan"
    LAUREN = "Lauren"
    JENNA = "Jenna"
    KAREN = "Karen"
    RICK = "Rick"
    BILL = "Bill"
    JAMES = "James"
    YANN = "Yann"
    EMILY = "Emily"
    ANNA = "Anna"
    JON = "Jon"
    BRENDA = "Brenda"
    BARBARA = "Barbara"

class OutputFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"

class TTSRequest(BaseModel):
    voice_id: SpeakerId
    output_format: OutputFormat
    text: str

app = FastAPI(
    title="Parler TTS API",
    description="Text-to-Speech API using Parler TTS model",
    version="1.0.0"
)

# Initialize model and tokenizer globally
print("Loading model and tokenizer...")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# Determine device_map based on availability
device_map_config = TORCH_DEVICE if torch.cuda.is_available() else None

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    # use_cache=True,
    device_map=device_map_config,
    low_cpu_mem_usage=torch.cuda.is_available(),\
).eval()

# Ensure model is on the correct device if device_map wasn't used (CPU case)
if not torch.cuda.is_available():
    model.to(TORCH_DEVICE)

# Enable torch.compile for better performance
# if hasattr(torch, 'compile') and torch.cuda.is_available():
#     model = torch.compile(model, mode="reduce-overhead")

# Get sampling rate from model config
# Handle potential AttributeError if audio_encoder doesn't exist or structure changes
try:
    SAMPLING_RATE = model.audio_encoder.config.sampling_rate
except AttributeError:
    print("Warning: Could not determine sampling rate from model.audio_encoder.config.sampling_rate. Defaulting to 24000.")
    # Fallback or check model documentation for correct sampling rate
    SAMPLING_RATE = 24000 # Common rate for ParlerTTS, adjust if needed

@app.post("/text_to_speech")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using specified voice and format"""
    try:
        # Construct description dynamically based on selected voice
        description = f"A voice description for {request.voice_id}. The speaker is speaking naturally and clearly."

        # Prepare inputs with optimized settings
        # Use context manager for device placement
        with torch.device(TORCH_DEVICE):
            encoded_inputs = tokenizer(
                description,
                return_tensors="pt",
                padding=True, # Consider 'max_length' if inputs can be very long
                truncation=True,
                max_length=1024 # Max length for description
            )

            prompt = tokenizer(
                request.text,
                return_tensors="pt",
                padding=True, # Consider 'max_length' if inputs can be very long
                truncation=True,
                max_length=1024 # Max length for text prompt
            )
        # --- Add Dynamic max_new_tokens ---
        # Estimate max tokens: Heuristic - adjust factor as needed
        # Assume roughly X audio tokens per input text token (e.g., 15-25 for ParlerTTS)
        # This needs tuning based on observed output lengths!
        estimated_max_tokens = len(prompt.input_ids[0]) * 25 # Example factor
        # Add a buffer and ensure a minimum
        calculated_max_new_tokens = max(150, estimated_max_tokens + 100)
        print(f"Setting max_new_tokens (estimated): {calculated_max_new_tokens}")

        # Optimized generation kwargs
        generation_kwargs = {
            "input_ids": encoded_inputs.input_ids,
            "attention_mask": encoded_inputs.attention_mask,
            "prompt_input_ids": prompt.input_ids,
            "prompt_attention_mask": prompt.attention_mask,
            "do_sample": True,
            "temperature": 0.7,
            "use_cache": True,
            "num_beams": 1,
            "max_new_tokens": calculated_max_new_tokens, # Use calculated value
            "length_penalty": 1.0,
            # "early_stopping": True, # Not useful with num_beams=1
            "repetition_penalty": 1.0,
            "pad_token_id": model.generation_config.pad_token_id, # Ensure pad_token_id is set
            "eos_token_id": model.generation_config.eos_token_id  # Ensure eos_token_id is set
        }

        # Generate audio with optimized settings
        # Use autocast only if on CUDA and using float16
        use_autocast = torch.cuda.is_available() and TORCH_DTYPE == torch.float16
        with torch.cuda.amp.autocast(enabled=use_autocast), torch.no_grad():
            output = model.generate(**generation_kwargs)

        # Ensure output is on CPU and float32 for processing
        audio_data = output.cpu().to(dtype=torch.float32).numpy().squeeze()

        # Normalize audio data safely
        max_abs_val = max(abs(audio_data.min()), abs(audio_data.max()))
        if max_abs_val > 0:
            audio_data = audio_data / max_abs_val
        else:
            print("Warning: Generated audio is silent.") # Handle silent audio case

        # Convert to bytes
        buffer = io.BytesIO()

        if request.output_format == OutputFormat.WAV:
            # Save as WAV
            sf.write(buffer, audio_data, SAMPLING_RATE, format='WAV', subtype='PCM_16')
            content_type = "audio/wav"
            buffer.seek(0)  # Reset buffer position for reading
        elif request.output_format == OutputFormat.MP3:
             # Convert WAV in memory to MP3
            # 1. Write the complete WAV data to the buffer first
            sf.write(buffer, audio_data, SAMPLING_RATE, format='WAV', subtype='PCM_16')
            buffer.seek(0) # Reset buffer position to the beginning

            # 2. Load the WAV data from the buffer into pydub
            try:
                audio_segment = pydub.AudioSegment.from_wav(buffer)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format='mp3')
                buffer = mp3_buffer # Replace buffer with the MP3 data
                content_type = "audio/mpeg"
                buffer.seek(0) # Reset buffer position for reading MP3 data
            except pydub.exceptions.CouldntDecodeError as pydub_err:
                 raise HTTPException(status_code=500, detail=f"Failed to convert WAV to MP3: {pydub_err}") from pydub_err
            except FileNotFoundError as ff_err:
                 # Handle missing ffmpeg/ffprobe
                 raise HTTPException(status_code=500, detail=f"Failed to convert to MP3. Ensure ffmpeg/ffprobe is installed and in PATH: {ff_err}") from ff_err
        else:
            # Should not happen due to Enum validation, but good practice
            raise HTTPException(status_code=400, detail="Invalid output format specified.")

        # buffer.seek(0)
        audio_bytes = buffer.getvalue()

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=audio.{request.output_format.value}" # Use .value
            }
        )

    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


if __name__ == "__main__":
    import uvicorn
    # Consider adding reload=False for production deployment
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) # Pass app as string for reload