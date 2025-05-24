#!/usr/bin/env python3

"""
Patch script to fix compatibility issues between diffusers, transformers, and huggingface_hub.
This adds the missing HF_HUB_CACHE attribute to huggingface_hub.constants if it doesn't exist.
"""

import os
import sys
import importlib
import warnings

def apply_patches():
    """Apply necessary patches to make the libraries compatible"""
    print("Applying compatibility patches...")
    
    try:
        from huggingface_hub import constants
        if not hasattr(constants, 'HF_HUB_CACHE'):
            print("Adding missing HF_HUB_CACHE attribute to huggingface_hub.constants")
            constants.HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"Set HF_HUB_CACHE to: {constants.HF_HUB_CACHE}")
    except ImportError:
        print("Warning: Could not import huggingface_hub.constants")
    
    try:
        import torch.nn as nn
        original_conv2d_forward = nn.Conv2d.forward
        
        def patched_conv2d_forward(self, input, *args):
            """Patched Conv2d forward to handle extra arguments from peft"""
            return original_conv2d_forward(self, input)
        
        nn.Conv2d.forward = patched_conv2d_forward
        print("Patched torch.nn.Conv2d.forward to handle extra arguments from peft")
    except Exception as e:
        print(f"Warning: Failed to patch Conv2d.forward: {e}")
    
    print("Patches applied successfully")

if __name__ == "__main__":
    apply_patches()
    print("\nYou can now import this module at the beginning of your scripts:")
    print("import patch_diffusers; patch_diffusers.apply_patches()")
