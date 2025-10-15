"""
Simple LLM Caption for ComfyUI
Created by: PixelaiLabs.com
Author: Aiconomist (@aiconomist on YouTube)
License: GPL-3.0

Automatic model downloading and setup - just clone and go!
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig
)
import folder_paths
import comfy.model_management
import torchvision.transforms.functional as TVF
from huggingface_hub import snapshot_download

print("=" * 80)
print("Simple LLM Caption for ComfyUI")
print("Created by: PixelaiLabs.com | Author: Aiconomist (@aiconomist)")
print("=" * 80)

# ============================================================================
# Joy Caption Image Adapter
# This is the key component that allows the LLM to "see" images
# ============================================================================

class ImageAdapter(nn.Module):
    """
    Image adapter that converts CLIP vision features to LLM embeddings.
    This allows the LLM to process visual information directly.
    Based on Joy Caption Alpha Two implementation.
    """
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, 
                 num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract
        
        if self.deep_extract:
            input_features = input_features * 5
        
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        
        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.cat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]
        
        x = self.ln1(x)
        
        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        
        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)
        
        return x
    
    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# Caption configuration (matching Joy Caption Alpha Two format)
# Each caption type has 3 templates:
# [0] = no length specified (for "any")
# [1] = numeric word count (for numbers like 50, 100)
# [2] = text length (for "short", "long", etc.)
CAPTION_TYPES = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone."
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone."
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image."
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image."
    ],
    "Booru Tags": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image."
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}."
    ],
    "Social Media": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post."
    ],
}

CAPTION_LENGTHS = [
    "any",
    "very short", 
    "short",
    "medium-length",  # Note: Joy Caption uses "medium-length" not "medium"
    "long",
    "very long",
]

# Joy Caption uses fixed max_new_tokens=300 for all caption lengths
# The length is controlled by the prompt text, not token limits
MAX_NEW_TOKENS = 300

def build_caption_prompt(caption_type, caption_length):
    """
    Build the caption prompt based on type and length.
    Matches Joy Caption Alpha Two's template selection logic.
    
    Returns the formatted prompt string.
    """
    # Get the template list for this caption type
    templates = CAPTION_TYPES[caption_type]
    
    # Determine which template to use based on caption_length
    if caption_length == "any" or caption_length is None:
        # Use template [0] - no length specified
        return templates[0]
    
    # Try to convert to int (for numeric word counts)
    try:
        word_count = int(caption_length)
        # Use template [1] - numeric word count
        return templates[1].format(word_count=word_count)
    except (ValueError, TypeError):
        # It's a text length like "short", "long", etc.
        # Use template [2] - text length
        return templates[2].format(length=caption_length)

def tensor_to_pil(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    # ComfyUI tensors are in format [B, H, W, C] with values 0-1
    numpy_image = tensor.cpu().numpy()
    if len(numpy_image.shape) == 4:
        numpy_image = numpy_image[0]  # Get first image if batch
    
    # Convert from 0-1 to 0-255
    numpy_image = (numpy_image * 255).astype(np.uint8)
    return Image.fromarray(numpy_image)

def modify_json_value(file_path, key, new_value):
    """Modify a specific value in a JSON file (Joy Caption helper function)"""
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    data[key] = new_value
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def download_model_from_hf(repo_id, model_name=None):
    """Auto-download model from Hugging Face"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub for automatic model downloads...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    
    llm_path = Path(folder_paths.models_dir) / "LLM"
    llm_path.mkdir(parents=True, exist_ok=True)
    
    if model_name is None:
        model_name = repo_id.split('/')[-1]
    
    local_dir = llm_path / model_name
    
    if local_dir.exists():
        print(f"Model already exists: {model_name}")
        return str(local_dir)
    
    print(f"\n{'='*80}")
    print(f"AUTO-DOWNLOADING MODEL: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"This will take a few minutes...")
    print(f"{'='*80}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"\n✅ Successfully downloaded {model_name}!")
        return str(local_dir)
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print(f"You can manually download from: https://huggingface.co/{repo_id}")
        return None

# Default recommended models  
DEFAULT_MODELS = {
    "Llama-3.1-8B-Lexi-Uncensored-V2-nf4": "John6666/Llama-3.1-8B-Lexi-Uncensored-V2-nf4",
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
}

def download_joy_caption_adapter():
    """
    Download the Joy Caption adapter models (EXACT Joy Caption process).
    Downloads:
    - image_adapter.pt (86 MB)
    - clip_model.pt (1.71 GB)
    - text_model/ folder with LoRA adapter (671 MB) - THIS IS KEY FOR CLEAN OUTPUT!
    """
    joy_caption_path = Path(folder_paths.models_dir) / "Joy_caption"
    joy_caption_path.mkdir(parents=True, exist_ok=True)
    
    # Check if all required files exist
    adapter_file = joy_caption_path / "image_adapter.pt"
    clip_model_file = joy_caption_path / "clip_model.pt"
    text_model_dir = joy_caption_path / "text_model"
    adapter_model_file = text_model_dir / "adapter_model.safetensors"
    
    if adapter_file.exists() and clip_model_file.exists() and adapter_model_file.exists():
        print(f"✅ Joy Caption adapter models already downloaded")
        return str(joy_caption_path)
    
    print(f"\n{'='*80}")
    print(f"AUTO-DOWNLOADING JOY CAPTION MODELS (Joy Caption Alpha Two)")
    print(f"Repository: fancyfeast/joy-caption-alpha-two")
    print(f"Destination: {joy_caption_path}")
    print(f"Total size: ~2.5 GB (image adapter + CLIP + LoRA adapter)")
    print(f"{'='*80}\n")
    
    # Files to download from cgrkzexw-599808 folder
    files_to_download = [
        "image_adapter.pt",
        "clip_model.pt",
        "text_model/README.md",
        "text_model/adapter_config.json",
        "text_model/adapter_model.safetensors",  # 671 MB - THE MAGIC!
        "text_model/special_tokens_map.json",
        "text_model/tokenizer.json",
        "text_model/tokenizer_config.json",
    ]
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Download attempt {attempt + 1}/{max_retries}...")
            from huggingface_hub import hf_hub_download
            import shutil
            
            # Download all files
            for filename in files_to_download:
                print(f"  Downloading {filename}...")
                hf_hub_download(
                    repo_id="fancyfeast/joy-caption-alpha-two",
                    filename=f"cgrkzexw-599808/{filename}",
                    repo_type="space",
                    local_dir=str(joy_caption_path.parent),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            
            # Move files from cgrkzexw-599808 to Joy_caption
            source_folder = joy_caption_path.parent / "cgrkzexw-599808"
            if source_folder.exists():
                # Move image_adapter.pt and clip_model.pt to root
                if (source_folder / "image_adapter.pt").exists():
                    shutil.move(str(source_folder / "image_adapter.pt"), str(adapter_file))
                if (source_folder / "clip_model.pt").exists():
                    shutil.move(str(source_folder / "clip_model.pt"), str(clip_model_file))
                
                # Move text_model folder
                if (source_folder / "text_model").exists():
                    if text_model_dir.exists():
                        shutil.rmtree(text_model_dir)
                    shutil.move(str(source_folder / "text_model"), str(text_model_dir))
                
                # Clean up source folder
                shutil.rmtree(source_folder)
            
            print(f"\n✅ Successfully downloaded Joy Caption models!")
            print(f"   📁 {adapter_file}")
            print(f"   📁 {clip_model_file}")
            print(f"   📁 {text_model_dir}")
            return str(joy_caption_path)
            
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                print(f"  ⏳ Retrying in {(attempt + 1) * 2} seconds...")
                time.sleep((attempt + 1) * 2)
            continue
    
    # All attempts failed - provide manual download instructions
    print(f"\n{'='*80}")
    print(f"❌ AUTOMATIC DOWNLOAD FAILED")
    print(f"{'='*80}")
    print(f"\n📖 MANUAL DOWNLOAD INSTRUCTIONS:")
    print(f"\n1. Go to: https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main/cgrkzexw-599808")
    print(f"\n2. Download these files:")
    print(f"   Root folder:")
    print(f"   - image_adapter.pt")
    print(f"   - clip_model.pt")
    print(f"\n   text_model folder (IMPORTANT - for clean captions):")
    print(f"   - adapter_config.json")
    print(f"   - adapter_model.safetensors (671 MB)")
    print(f"   - special_tokens_map.json")
    print(f"   - tokenizer.json")
    print(f"   - tokenizer_config.json")
    print(f"\n3. Save them to: {joy_caption_path}")
    print(f"\n   Final structure:")
    print(f"   {joy_caption_path}/")
    print(f"   ├── image_adapter.pt")
    print(f"   ├── clip_model.pt")
    print(f"   └── text_model/")
    print(f"       ├── adapter_config.json")
    print(f"       ├── adapter_model.safetensors")
    print(f"       └── tokenizer files...")
    print(f"\n4. Restart ComfyUI")
    print(f"\n{'='*80}\n")
    
    return None

# Common visual concepts for CLIP-based image understanding
VISUAL_CONCEPTS = [
    # Human-related (MOST IMPORTANT - check these first)
    "a woman", "a man", "a person", "people", "a child",
    "woman's face", "man's face", "human face", "portrait",
    "female person", "male person", "human portrait",
    
    # Photo types
    "professional photo", "amateur photo", "artistic",
    "a photo", "a painting", "a drawing", "a digital art", "a 3D render",
    
    # Settings  
    "indoor scene", "outdoor scene", "landscape", "cityscape",
    "close-up", "wide shot", "aerial view",
    "daytime", "nighttime", "sunset", "sunrise",
    
    # Style
    "colorful", "black and white", "vibrant", "muted colors",
    
    # Animals & Objects (LAST - lowest priority)
    "an animal", "a dog", "a cat", "a bird", "a pet",
]

def process_caption_text(text, gender_age_replacement="", hair_replacement="", 
                        body_size_replacement="", lora_trigger="", 
                        remove_tattoos=False, remove_jewelry=False):
    """
    Process caption text with various replacements and cleanups.
    Integrated from Text Processor by Aiconomist.
    
    If all parameters are empty/default, returns original text unchanged.
    """
    
    # Check if any processing is requested
    has_processing = (gender_age_replacement or hair_replacement or body_size_replacement or 
                     lora_trigger or remove_tattoos or remove_jewelry)
    
    if not has_processing:
        return text  # Return original if no processing requested
    
    # Debug: Show original caption
    print(f"📝 Original caption: {text}")
    
    # Remove surrounding quotes if the entire caption is wrapped in quotes
    # (Some LLMs wrap their output in quotes)
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
        print(f"   Removed surrounding quotes")
    
    # Build unwanted tags list
    unwanted_tags = []
    
    # Add jewelry patterns if remove_jewelry is True
    if remove_jewelry:
        unwanted_tags.extend([
            r"\bjewelry\b", r"\brings\b", r"\bpiercing\b", r"\bring\b",
            r"\bearrings\b", r"\bnecklace\b", r"\bbracelet\b",
            r"\bwatch\b", r"\bangklet\b", r"\bbody jewelry\b", r"\bnose ring\b",
            r"\bear piercing\b", r"\blip piercing\b", r"\btongue piercing\b"
        ])
    
    # Add tattoo patterns if remove_tattoos is True
    if remove_tattoos:
        unwanted_tags.extend([
            r"\btattoo\b", r"\btattoos\b", r"\btattooed\b", r"\btattooing\b",
            r"\bink\b", r"\binked\b", r"\bbody art\b", r"\btribal tattoo\b",
            r"\bsleeve tattoo\b", r"\bface tattoo\b", r"\bneck tattoo\b",
            r"\bback tattoo\b", r"\bchest tattoo\b", r"\barm tattoo\b",
            r"\bleg tattoo\b", r"\bshoulder tattoo\b", r"\bwrist tattoo\b",
            r"\bankle tattoo\b", r"\bfinger tattoo\b", r"\btattoo sleeve\b"
        ])
    
    # Remove unwanted tags
    for pattern in unwanted_tags:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Replace hair keywords
    if hair_replacement:
        hair_patterns = [
            # Compound hair descriptions (match longest first)
            r"\bdark\s+brown\s+hair\b", r"\blight\s+brown\s+hair\b",
            r"\bdark\s+blonde\s+hair\b", r"\blight\s+blonde\s+hair\b",
            r"\blong\s+hair\b", r"\bshort\s+hair\b", r"\bmedium\s+hair\b",
            # Simple colors
            r"\bbrown\s+hair\b", r"\bblonde\s+hair\b", r"\bred\s+hair\b",
            r"\bblack\s+hair\b", r"\bwhite\s+hair\b", r"\bgray\s+hair\b",
            r"\bsilver\s+hair\b", r"\bpink\s+hair\b", r"\bblue\s+hair\b",
            # Standalone descriptors
            r"\bblonde\b", r"\brunette\b", r"\bredhead\b",
            # Hairstyles
            r"\bponytail\b", r"\bbun\b", r"\bbraids\b", r"\bbangs\b"
        ]
        for pattern in hair_patterns:
            text = re.sub(pattern, hair_replacement, text, flags=re.IGNORECASE)
    
    # Replace body size keywords
    if body_size_replacement:
        body_size_patterns = [
            r"\bthin\b", r"\bslim\b", r"\bskinny\b", r"\bpetite\b",
            r"\bcurvy\b", r"\bvoluptuous\b", r"\bplus size\b",
            r"\bmuscular\b", r"\btoned\b", r"\bfit\b", r"\bathletic\b"
        ]
        for pattern in body_size_patterns:
            text = re.sub(pattern, body_size_replacement, text, flags=re.IGNORECASE)
    
    # Replace gender/age keywords
    if gender_age_replacement:
        gender_patterns = [
            r"\bblonde woman\b", r"\bredhead woman\b", r"\bbrunette woman\b",
            r"\b1girl\b", r"\bwoman\b", r"\bman\b", r"\bgirl\b", r"\bboy\b",
            r"\blady\b", r"\bgentleman\b", r"\bmale\b", r"\bfemale\b",
            r"\byoung woman\b", r"\byoung man\b"
        ]
        for pattern in gender_patterns:
            text = re.sub(pattern, gender_age_replacement, text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and commas FIRST (before adding lora trigger)
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r',\s*,+', ',', text)  # Multiple commas to single
    text = re.sub(r'^\s*,\s*', '', text)  # Leading comma
    text = re.sub(r'\s*,\s*$', '', text)  # Trailing comma
    text = text.strip()
    
    # Add LoRA trigger at the beginning AFTER cleanup
    if lora_trigger and text:  # Only add if there's text remaining
        text = f"{lora_trigger.strip()}, {text}"
    elif lora_trigger and not text:  # If no text left, just use trigger
        text = lora_trigger.strip()
    
    # Debug: Show processed caption
    print(f"✅ Processed caption: {text}")
    
    return text

def get_available_llm_models():
    """
    Get list of Joy Caption-compatible LLM models.
    Always shows AUTO-DOWNLOAD options first, then scans for Llama models.
    """
    llm_path = Path(folder_paths.models_dir) / "LLM"
    if not llm_path.exists():
        llm_path.mkdir(parents=True, exist_ok=True)
    
    # Always show AUTO-DOWNLOAD options first (recommended)
    models = ["AUTO-DOWNLOAD: " + name for name in DEFAULT_MODELS.keys()]
    
    # Scan for existing Llama-based models (Joy Caption compatible)
    compatible_keywords = ['llama', 'lexi', 'meta-llama']  # Joy Caption uses Llama models
    incompatible_keywords = ['florence', 'clip', 'siglip', 'bert', 'vit']  # Vision/other models
    
    for item in llm_path.iterdir():
        if item.is_dir():
            model_name_lower = item.name.lower()
            
            # Check if it's a valid model directory
            if not ((item / "config.json").exists() or (item / "adapter_config.json").exists()):
                continue
            
            # Skip incompatible models (Florence, CLIP, etc.)
            if any(keyword in model_name_lower for keyword in incompatible_keywords):
                continue
            
            # Only include Llama-based models (Joy Caption compatible)
            if any(keyword in model_name_lower for keyword in compatible_keywords):
                if item.name not in models:  # Avoid duplicates
                    models.append(item.name)
    
    return models

class SimpleLLMCaptionLoader:
    """Node to load the LLM and vision models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.clip_model = None  # SigLIP vision model for Joy Caption
        self.clip_processor = None
        self.clip_tokenizer = None
        self.image_adapter = None  # Joy Caption image adapter
        self.current_model_name = None
        self.device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (get_available_llm_models(),),
                "use_4bit": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LLM_PIPELINE",)
    FUNCTION = "load_models"
    CATEGORY = "image/captioning"
    
    def load_models(self, llm_model, use_4bit):
        """Load the LLM and vision models - auto-downloads if needed"""
        
        # Handle auto-download option
        if llm_model.startswith("AUTO-DOWNLOAD: "):
            model_name = llm_model.replace("AUTO-DOWNLOAD: ", "")
            repo_id = DEFAULT_MODELS.get(model_name)
            
            if repo_id:
                print(f"\n🚀 First-time setup: Downloading {model_name}...")
                print(f"This is a one-time download (~4-5GB)")
                print(f"Future loads will be instant!\n")
                
                downloaded_path = download_model_from_hf(repo_id, model_name)
                if downloaded_path:
                    llm_model = model_name
                else:
                    raise ValueError(f"Failed to download {model_name}. Check your internet connection.")
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        # === STEP 1: Download Joy Caption adapter FIRST ===
        print(f"\n🔧 Checking Joy Caption adapter...")
        joy_caption_path = download_joy_caption_adapter()
        if not joy_caption_path:
            raise ValueError("Joy Caption adapter is required. Please check download instructions above.")
        
        text_model_path = Path(joy_caption_path) / "text_model"
        if not text_model_path.exists():
            raise ValueError(f"Joy Caption text_model folder not found at {text_model_path}")
        
        # Only reload if model changed
        if self.current_model_name != llm_model or self.model is None:
            print(f"Loading LLM with Joy Caption LoRA adapter: {llm_model}")
            
            # Clear previous model
            if self.model is not None:
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()
            
            llm_path = Path(folder_paths.models_dir) / "LLM" / llm_model
            
            # === JOY CAPTION PROCESS: Load tokenizer from text_model folder ===
            print(f"📖 Loading Joy Caption tokenizer from text_model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(text_model_path),
                use_fast=True
            )
            assert isinstance(self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(self.tokenizer)}"
            
            # === JOY CAPTION PROCESS: Update adapter_config.json to point to base LLM ===
            adapter_config_path = text_model_path / "adapter_config.json"
            if adapter_config_path.exists():
                print(f"🔧 Updating adapter config to point to base LLM...")
                modify_json_value(str(adapter_config_path), "base_model_name_or_path", str(llm_path))
            
            # === JOY CAPTION PROCESS: Load LLM with LoRA adapter ===
            # The text_model folder contains adapter_model.safetensors (671MB LoRA)
            # This adapter makes the LLM output clean captions without conversational prefixes!
            print(f"🚀 Loading base LLM with Joy Caption LoRA adapter...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(text_model_path),  # Load from text_model (has adapter)
                device_map=self.device,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=comfy.model_management.text_encoder_dtype()
            )
            
            self.model.eval()
            self.current_model_name = llm_model
            print(f"✅ LLM with Joy Caption LoRA adapter loaded successfully!")
        
        # Load SigLIP vision model (required for Joy Caption)
        if self.clip_model is None:
            print(f"\n🔧 Loading SigLIP vision model for Joy Caption...")
            siglip_model_name = "google/siglip-so400m-patch14-384"
            siglip_path = Path(folder_paths.models_dir) / "clip" / "siglip-so400m-patch14-384"
            
            if not siglip_path.exists():
                print(f"📥 First-time download: SigLIP (~1.5GB)")
                print(f"This is required for Joy Caption. Please wait...")
                try:
                    from transformers import AutoModel, AutoProcessor
                    # Download SigLIP
                    siglip_full = AutoModel.from_pretrained(siglip_model_name, trust_remote_code=True)
                    siglip_processor = AutoProcessor.from_pretrained(siglip_model_name, trust_remote_code=True)
                    # Save locally
                    siglip_path.mkdir(parents=True, exist_ok=True)
                    siglip_full.save_pretrained(str(siglip_path))
                    siglip_processor.save_pretrained(str(siglip_path))
                    print(f"✅ SigLIP downloaded successfully!")
                except Exception as e:
                    print(f"❌ Error downloading SigLIP: {e}")
                    raise ValueError(f"Failed to download SigLIP. Check your internet connection.")
            
            # Load SigLIP vision model
            from transformers import AutoModel
            siglip_full = AutoModel.from_pretrained(str(siglip_path), trust_remote_code=True, local_files_only=True)
            self.clip_model = siglip_full.vision_model
            self.clip_model.to(self.device)
            self.clip_model.eval()
            print(f"✅ SigLIP vision model loaded")
            
            # Load custom CLIP weights from Joy Caption
            clip_model_file = Path(joy_caption_path) / "clip_model.pt"
            if clip_model_file.exists():
                print(f"🔧 Loading Joy Caption custom CLIP weights...")
                checkpoint = torch.load(clip_model_file, map_location='cpu', weights_only=True)
                checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
                self.clip_model.load_state_dict(checkpoint)
                del checkpoint
                print(f"✅ Joy Caption CLIP weights loaded")
            else:
                print(f"⚠️ Warning: clip_model.pt not found at {clip_model_file}")
        
        # Load Joy Caption image adapter
        if self.image_adapter is None:
            print(f"\n🔧 Loading Joy Caption Image Adapter...")
            adapter_file = Path(joy_caption_path) / "image_adapter.pt"
            if adapter_file.exists():
                # Create image adapter (SigLIP hidden size = 1152, Llama 3.1 = 4096)
                self.image_adapter = ImageAdapter(
                    input_features=1152,
                    output_features=4096,
                    ln1=False,
                    pos_emb=False,
                    num_image_tokens=38,
                    deep_extract=False
                )
                # Load weights
                self.image_adapter.load_state_dict(torch.load(adapter_file, map_location=self.device, weights_only=True))
                self.image_adapter.to(self.device)
                self.image_adapter.eval()
                print(f"✅ Joy Caption Image Adapter loaded successfully!")
            else:
                print(f"⚠️ Warning: image_adapter.pt not found at {adapter_file}")
                self.image_adapter = None
        
        # Return pipeline object (cleaned up - no unused CLIP components)
        pipeline = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "clip_model": self.clip_model,  # SigLIP vision model
            "image_adapter": self.image_adapter,  # Joy Caption adapter
            "device": self.device,
        }
        
        return (pipeline,)

class SimpleLLMCaption:
    """Node to generate captions from images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LLM_PIPELINE",),
                "image": ("IMAGE",),
                "caption_type": (list(CAPTION_TYPES.keys()),),
                "caption_length": (CAPTION_LENGTHS, {"default": "medium-length"}),
            },
            "optional": {
                "lora_trigger": ("STRING", {"default": "", "multiline": False}),
                "gender_age_replacement": ("STRING", {"default": "", "multiline": False}),
                "hair_replacement": ("STRING", {"default": "", "multiline": False}),
                "body_size_replacement": ("STRING", {"default": "", "multiline": False}),
                "remove_tattoos": ("BOOLEAN", {"default": False}),
                "remove_jewelry": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"
    CATEGORY = "image/captioning"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when any parameter changes (prevent caching issues)
        import hashlib
        import json
        # Create hash of all parameters to detect changes
        param_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def generate_caption(self, pipeline, image, caption_type, caption_length,
                        lora_trigger="", gender_age_replacement="", hair_replacement="", 
                        body_size_replacement="", remove_tattoos=False, remove_jewelry=False):
        """Generate caption for the input image with optional text processing"""
        
        model = pipeline["model"]
        tokenizer = pipeline["tokenizer"]
        clip_model = pipeline["clip_model"]
        image_adapter = pipeline.get("image_adapter")
        device = pipeline["device"]
        
        # Convert tensor to PIL and resize for SigLIP (384x384)
        pil_image = tensor_to_pil(image)
        pil_image = pil_image.resize((384, 384), Image.LANCZOS)
        
        # Check if we have the image adapter for full Joy Caption functionality
        if image_adapter is not None:
            print("🎨 Using Joy Caption Image Adapter (NSFW-capable)")
            
            # Preprocess image for SigLIP
            pixel_values = TVF.pil_to_tensor(pil_image).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(device)
            
            # Get vision features from SigLIP (Joy Caption process)
            with torch.no_grad():
                vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
                
                # Check output type and extract hidden states correctly
                if hasattr(vision_outputs, 'hidden_states') and vision_outputs.hidden_states is not None:
                    # Output is a model output object with hidden_states
                    hidden_states = vision_outputs.hidden_states
                elif isinstance(vision_outputs, tuple):
                    # Output is a tuple (last_hidden_state, pooler_output, hidden_states)
                    hidden_states = vision_outputs[2] if len(vision_outputs) > 2 else vision_outputs
                else:
                    # Output is just the hidden states tensor
                    hidden_states = vision_outputs
                
                # Use image adapter to convert vision features to LLM embeddings
                embedded_images = image_adapter(hidden_states)
            
            # Build prompt based on caption type and length (Joy Caption style)
            prompt_str = build_caption_prompt(caption_type, caption_length)
            
            # Build the conversation (Joy Caption style)
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt_str},
            ]
            
            # Format the conversation
            convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            
            # Tokenize the conversation
            convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
            prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
            convo_tokens = convo_tokens.squeeze(0)
            prompt_tokens = prompt_tokens.squeeze(0)
            
            # Calculate where to inject the image
            eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
            preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]
            
            # Embed the tokens
            convo_embeds = model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))
            
            # Construct the input with image embeddings
            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],  # Part before the prompt
                embedded_images.to(dtype=convo_embeds.dtype),  # Image embeddings
                convo_embeds[:, preamble_len:],  # The prompt and anything after it
            ], dim=1).to(device)
            
            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),  # Dummy tokens for the image
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate caption (Joy Caption settings: max_tokens=300, temp=0.6, top_p=0.9)
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids,
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    suppress_tokens=None,
                )
            
            # Trim off the prompt
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]
            
            # Decode (Joy Caption LoRA adapter ensures clean output - no cleaning needed!)
            caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            caption = caption.strip()
            
        else:
            # Fallback mode without image adapter (WARNING: limited functionality)
            print("⚠️ WARNING: Image adapter not available!")
            print("   Please download Joy Caption adapter for full NSFW functionality")
            print("   Manual download: See console instructions above")
            return ("ERROR: Joy Caption adapter required. See console for download instructions.",)
        
        # Apply text processing (integrated from Text Processor by Aiconomist)
        caption = process_caption_text(
            caption, 
            gender_age_replacement=gender_age_replacement,
            hair_replacement=hair_replacement,
            body_size_replacement=body_size_replacement,
            lora_trigger=lora_trigger,
            remove_tattoos=remove_tattoos,
            remove_jewelry=remove_jewelry
        )
        
        # Offload models to CPU to free VRAM
        comfy.model_management.soft_empty_cache()
        
        return (caption.strip(),)

class SimpleLLMCaptionAdvanced:
    """Advanced node with custom prompts and more options"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LLM_PIPELINE",),
                "image": ("IMAGE",),
                "caption_type": (list(CAPTION_TYPES.keys()),),
                "caption_length": (CAPTION_LENGTHS, {"default": "medium"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 300, "min": 50, "max": 1000, "step": 50}),
            },
            "optional": {
                "append_to_caption": ("STRING", {"multiline": True, "default": "", "placeholder": "Text to add at the end of caption..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Your negative prompt here..."}),
                "lora_trigger": ("STRING", {"default": "", "multiline": False}),
                "gender_age_replacement": ("STRING", {"default": "", "multiline": False}),
                "hair_replacement": ("STRING", {"default": "", "multiline": False}),
                "body_size_replacement": ("STRING", {"default": "", "multiline": False}),
                "remove_tattoos": ("BOOLEAN", {"default": False}),
                "remove_jewelry": ("BOOLEAN", {"default": False}),
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "generate_caption"
    CATEGORY = "image/captioning"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when any parameter changes (prevent caching issues)
        import hashlib
        import json
        param_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def generate_caption(self, pipeline, image, caption_type, caption_length, temperature, top_p, max_new_tokens, 
                        append_to_caption="", negative_prompt="", lora_trigger="", gender_age_replacement="", hair_replacement="", 
                        body_size_replacement="", remove_tattoos=False, remove_jewelry=False, prefix="", suffix=""):
        """Generate caption with advanced options and text processing"""
        
        model = pipeline["model"]
        tokenizer = pipeline["tokenizer"]
        clip_model = pipeline["clip_model"]
        image_adapter = pipeline.get("image_adapter")
        device = pipeline["device"]
        
        # Convert tensor to PIL and resize for SigLIP
        pil_image = tensor_to_pil(image)
        pil_image = pil_image.resize((384, 384), Image.LANCZOS)
        
        # Use Joy Caption image adapter if available
        print("🔍 [Advanced] Processing image...")
        
        if image_adapter is not None:
            print("🎨 [Advanced] Using Joy Caption Image Adapter (NSFW-capable)")
            
            # Preprocess image for SigLIP
            pixel_values = TVF.pil_to_tensor(pil_image).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(device)
            
            # Get vision features and embed (Joy Caption process)
            with torch.no_grad():
                vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
                
                # Extract hidden states correctly
                if hasattr(vision_outputs, 'hidden_states') and vision_outputs.hidden_states is not None:
                    hidden_states = vision_outputs.hidden_states
                elif isinstance(vision_outputs, tuple):
                    hidden_states = vision_outputs[2] if len(vision_outputs) > 2 else vision_outputs
                else:
                    hidden_states = vision_outputs
                
                embedded_images = image_adapter(hidden_states)
        
        # Build prompt using Joy Caption template
        prompt_str = build_caption_prompt(caption_type, caption_length)
        
        # Create conversation (Joy Caption style)
        conversation = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt_str}
        ]
        
        # Generate caption
        if image_adapter is not None:
            # Use image embeddings (Joy Caption style)
            convo_string = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
            prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
            convo_tokens = convo_tokens.squeeze(0)
            prompt_tokens = prompt_tokens.squeeze(0)
            
            # Calculate where to inject the image
            eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
            preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]
            
            # Embed tokens and construct input with image
            convo_embeds = model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))
            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],
                embedded_images.to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:],
            ], dim=1).to(device)
            
            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate caption (Advanced allows custom temp/top_p but uses Joy Caption's max_tokens=300)
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids,
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,  # Advanced node allows custom control
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    suppress_tokens=None,
                )
            
            # Trim and decode (Joy Caption LoRA adapter ensures clean output!)
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]
            
            caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            caption = caption.strip()
            
        else:
            # Fallback mode without image adapter
            print("⚠️ WARNING: Image adapter not available!")
            return ("ERROR: Joy Caption adapter required. See console for download instructions.",)
        
        # Apply text processing (integrated from Text Processor by Aiconomist)
        caption = process_caption_text(
            caption, 
            gender_age_replacement=gender_age_replacement,
            hair_replacement=hair_replacement,
            body_size_replacement=body_size_replacement,
            lora_trigger=lora_trigger,
            remove_tattoos=remove_tattoos,
            remove_jewelry=remove_jewelry
        )
        
        # Build final positive prompt
        positive_prompt = caption
        
        # Add prefix/suffix (applied after text processing)
        if prefix:
            positive_prompt = f"{prefix} {positive_prompt}"
        if suffix:
            positive_prompt = f"{positive_prompt} {suffix}"
        
        # Add append_to_caption at the END (user's custom text)
        if append_to_caption.strip():
            positive_prompt = f"{positive_prompt}, {append_to_caption.strip()}"
        
        # Prepare negative prompt (user provides this manually)
        negative_prompt_text = negative_prompt.strip() if negative_prompt else ""
        
        # Offload models to CPU to free VRAM
        comfy.model_management.soft_empty_cache()
        
        return (positive_prompt.strip(), negative_prompt_text)

class SimpleLLMCaptionBatch:
    """Batch process multiple images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LLM_PIPELINE",),
                "input_directory": ("STRING", {"default": ""}),
                "output_directory": ("STRING", {"default": ""}),
                "caption_type": (list(CAPTION_TYPES.keys()),),
                "caption_length": (CAPTION_LENGTHS, {"default": "medium"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 300, "min": 50, "max": 1000, "step": 50}),
                "save_as_txt": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_trigger": ("STRING", {"default": "", "multiline": False}),
                "gender_age_replacement": ("STRING", {"default": "", "multiline": False}),
                "hair_replacement": ("STRING", {"default": "", "multiline": False}),
                "body_size_replacement": ("STRING", {"default": "", "multiline": False}),
                "remove_tattoos": ("BOOLEAN", {"default": False}),
                "remove_jewelry": ("BOOLEAN", {"default": False}),
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "batch_caption"
    CATEGORY = "image/captioning"
    OUTPUT_NODE = True
    
    def batch_caption(self, pipeline, input_directory, output_directory, caption_type, caption_length, temperature, max_new_tokens, save_as_txt,
                     lora_trigger="", gender_age_replacement="", hair_replacement="", body_size_replacement="",
                     remove_tattoos=False, remove_jewelry=False, prefix="", suffix=""):
        """
        Process all images in a directory with Joy Caption adapter.
        Saves images and captions with chronological numbering for LoRA/model training.
        
        Example:
        Input:  image1.jpg, image2.png, image3.jpg
        Output: 1.png, 1.txt, 2.png, 2.txt, 3.png, 3.txt
        """
        
        if not input_directory or not os.path.exists(input_directory):
            return ("Error: Input directory does not exist",)
        
        # Output directory is required for batch processing
        if not output_directory:
            return ("Error: Please specify an output directory for chronological naming",)
        
        os.makedirs(output_directory, exist_ok=True)
        
        model = pipeline["model"]
        tokenizer = pipeline["tokenizer"]
        clip_model = pipeline["clip_model"]
        image_adapter = pipeline.get("image_adapter")
        device = pipeline["device"]
        
        # Check if Joy Caption adapter is available
        if image_adapter is None:
            return ("Error: Joy Caption adapter required for batch processing. See console for download instructions.",)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        # Find all images
        input_path = Path(input_directory)
        image_files = []
        
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        # Remove duplicates (in case of case-insensitive filesystems) and sort
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            return (f"No images found in {input_directory}",)
        
        # Build prompt using Joy Caption template
        prompt_str = build_caption_prompt(caption_type, caption_length)
        
        processed = 0
        errors = 0
        
        print(f"\n{'='*60}")
        print(f"📸 Batch Captioning for LoRA/Model Training")
        print(f"   Input:  {len(image_files)} images")
        print(f"   Output: Chronological numbering (1.png, 2.png, ...)")
        print(f"{'='*60}\n")
        
        # Process each image with chronological numbering
        for index, img_path in enumerate(image_files, start=1):
            try:
                print(f"[{index}/{len(image_files)}] Processing: {img_path.name}")
                
                # Load and convert image to RGB
                pil_image = Image.open(img_path)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                
                # Resize for SigLIP (384x384)
                pil_image_resized = pil_image.resize((384, 384), Image.LANCZOS)
                
                # Preprocess image for SigLIP (Joy Caption process)
                pixel_values = TVF.pil_to_tensor(pil_image_resized).unsqueeze(0) / 255.0
                pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
                pixel_values = pixel_values.to(device)
                
                # Get vision features and embed (Joy Caption process)
                with torch.no_grad():
                    vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
                    
                    # Extract hidden states correctly
                    if hasattr(vision_outputs, 'hidden_states') and vision_outputs.hidden_states is not None:
                        hidden_states = vision_outputs.hidden_states
                    elif isinstance(vision_outputs, tuple):
                        hidden_states = vision_outputs[2] if len(vision_outputs) > 2 else vision_outputs
                    else:
                        hidden_states = vision_outputs
                    
                    embedded_images = image_adapter(hidden_states)
                
                # Create conversation (Joy Caption style)
                conversation = [
                    {"role": "system", "content": "You are a helpful image captioner."},
                    {"role": "user", "content": prompt_str}
                ]
                
                # Format conversation (Joy Caption process)
                convo_string = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                assert isinstance(convo_string, str)
                
                # Tokenize
                convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
                prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
                assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
                convo_tokens = convo_tokens.squeeze(0)
                prompt_tokens = prompt_tokens.squeeze(0)
                
                # Calculate where to inject the image
                eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
                assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"
                
                preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]
                
                # Embed the tokens
                convo_embeds = model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))
                
                # Construct the input (Joy Caption process)
                input_embeds = torch.cat([
                    convo_embeds[:, :preamble_len],
                    embedded_images.to(dtype=convo_embeds.dtype),
                    convo_embeds[:, preamble_len:],
                ], dim=1).to(device)
                
                input_ids = torch.cat([
                    convo_tokens[:preamble_len].unsqueeze(0),
                    torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                    convo_tokens[preamble_len:].unsqueeze(0),
                ], dim=1).to(device)
                attention_mask = torch.ones_like(input_ids)
                
                # Generate caption (Joy Caption settings)
                with torch.no_grad():
                    generate_ids = model.generate(
                        input_ids,
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        suppress_tokens=None,
                    )
                
                # Trim and decode (Joy Caption process)
                generate_ids = generate_ids[:, input_ids.shape[1]:]
                if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
                    generate_ids = generate_ids[:, :-1]
                
                caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                caption = caption.strip()
                
                # Apply text processing
                caption = process_caption_text(
                    caption, 
                    gender_age_replacement=gender_age_replacement,
                    hair_replacement=hair_replacement,
                    body_size_replacement=body_size_replacement,
                    lora_trigger=lora_trigger,
                    remove_tattoos=remove_tattoos,
                    remove_jewelry=remove_jewelry
                )
                
                # Add prefix/suffix
                if prefix:
                    caption = f"{prefix} {caption}"
                if suffix:
                    caption = f"{caption} {suffix}"
                
                # Save with chronological numbering for LoRA/model training
                output_image_path = Path(output_directory) / f"{index}.png"
                output_txt_path = Path(output_directory) / f"{index}.txt"
                
                # Save image as PNG (standard format for training)
                pil_image.save(output_image_path, format='PNG')
                
                # Save caption
                if save_as_txt:
                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                
                processed += 1
                print(f"  ✅ Saved: {index}.png + {index}.txt")
                print(f"     Caption: {caption[:80]}...")
                
            except Exception as e:
                errors += 1
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Offload models to free VRAM
        comfy.model_management.soft_empty_cache()
        
        result = f"✅ Batch complete!\n   Processed: {processed} images\n   Errors: {errors}\n   Output: {output_directory}"
        print(f"\n{'='*60}")
        print(result)
        print(f"{'='*60}\n")
        
        return (result,)

# Clean up on module unload
def cleanup():
    torch.cuda.empty_cache()

