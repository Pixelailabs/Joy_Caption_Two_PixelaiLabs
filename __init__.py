"""
Joy Caption Two - PixelaiLabs Edition
A fully automated ComfyUI custom node for image captioning using Joy Caption Alpha Two.

Created by: PixelaiLabs.com
Author: Aiconomist (@aiconomist on YouTube)
License: GPL-3.0

Based on Joy Caption Alpha Two:
- Original Joy Caption by fpgaminer: https://github.com/fpgaminer/joycaption
- Joy Caption Alpha Two by fancyfeast: https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two
- Implementation reference: ComfyUI_SLK_joy_caption_two by EvilBT

This is a derivative work licensed under GPL-3.0
"""

from .simple_joy_caption import (
    SimpleLLMCaptionLoader, 
    SimpleLLMCaption,
    SimpleLLMCaptionAdvanced,
    SimpleLLMCaptionBatch
)

NODE_CLASS_MAPPINGS = {
    "SimpleLLMCaptionLoader": SimpleLLMCaptionLoader,
    "SimpleLLMCaption": SimpleLLMCaption,
    "SimpleLLMCaptionAdvanced": SimpleLLMCaptionAdvanced,
    "SimpleLLMCaptionBatch": SimpleLLMCaptionBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleLLMCaptionLoader": "Simple LLM Caption Loader",
    "SimpleLLMCaption": "Simple LLM Caption",
    "SimpleLLMCaptionAdvanced": "Simple LLM Caption (Advanced)",
    "SimpleLLMCaptionBatch": "Simple LLM Caption (Batch)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("\n" + "="*60)
print("✅ Joy Caption Two - PixelaiLabs Edition")
print("   Created by PixelaiLabs.com | @aiconomist")
print("   Based on Joy Caption Alpha Two (GPL-3.0)")
print("   Models download automatically on first use")
print("="*60 + "\n")

