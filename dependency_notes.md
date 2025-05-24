# Dependency Compatibility Issues
The following package combinations were tested but resulted in import errors:
- diffusers 0.21.4 + huggingface-hub 0.13.2 + transformers 4.36.0
- diffusers 0.21.4 + huggingface-hub 0.32.0 + transformers 4.30.0

Error: module 'huggingface_hub.constants' has no attribute 'HF_HUB_CACHE'

This appears to be a compatibility issue between these packages. Further investigation is needed to find a working combination.
