# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI-powered floor plan generation system that learns from architectural PDF drawings to generate residential floor plans using Stable Diffusion with LoRA fine-tuning. The system outputs 2D SVG plans and editable 3D FreeCAD models based on the Japanese 910mm/455mm grid system.

## Key Commands

### Setup and Environment
```bash
# Create and activate virtual environment
python3.11 -m venv floorplan_env
source floorplan_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize directory structure
./setup_dirs.sh
```

### Development Workflows

**Process PDFs and prepare training data:**
```bash
python scripts/process_pdfs.py
python scripts/prepare_training_data.py --pdf_dir data/raw_pdfs --output_dir data/training
```

**Train the model:**
```bash
python scripts/train_model.py --data_dir data/training --epochs 20
```

**Run full pipeline (training + UI):**
```bash
python scripts/train_and_display.py
# Skip training and just launch UI:
python scripts/train_and_display.py --skip-training
```

**Launch Streamlit UI directly:**
```bash
streamlit run src/ui/main_app.py --server.port 8501
```

**Generate a single plan:**
```bash
python scripts/generate_plan.py --width 11 --height 10 --output outputs/
```

**Run tests:**
```bash
pytest                    # All tests
pytest tests/unit/       # Unit tests only
pytest --cov=src         # With coverage
python scripts/performance_test.py  # Performance benchmark
```

**Code quality:**
```bash
black .                   # Format code
isort .                   # Sort imports
flake8 .                  # Lint code
mypy src/                 # Type checking
```

## High-Level Architecture

### Pipeline Flow
```
PDF Input → OCR Extraction → Grid Normalization → Training Data Generation
                                                           ↓
UI Input → Site Mask → AI Generation ← LoRA Fine-tuned Model
                            ↓
                   Constraint Validation
                            ↓
                    SVG/FreeCAD Output
```

### Key Components

1. **PDF Processing Pipeline** (`src/preprocessing/`)
   - `dimension_extractor.py`: OCR-based dimension extraction using PaddleOCR/EasyOCR
   - `grid_normalizer.py`: Converts architectural drawings to 910mm grid system
   - `training_data_generator.py`: Creates AI training pairs with site masks

2. **AI Architecture** (`src/training/`, `src/inference/`)
   - Base model: Stable Diffusion v1.4 (lightweight, open-access)
   - Fine-tuning: LoRA with rank 64 for efficient training
   - Custom `LoRATrainer` handles training with MPS support for M4 Max
   - `FloorPlanGenerator` wraps Img2Img pipeline for inference

3. **Constraint System** (`src/constraints/`)
   - Rule-based validation for architectural constraints
   - OR-Tools CP-SAT solver framework (prepared for future enhancements)
   - Checks wall continuity, room connectivity, and building codes

4. **CAD Integration** (`src/freecad_bridge/`)
   - `FreeCADGenerator` creates editable 3D models from grid plans
   - Converts grid coordinates to real-world dimensions
   - Gracefully handles FreeCAD unavailability

5. **UI Layer** (`src/ui/`)
   - Streamlit-based web interface
   - Session state management for workflow persistence
   - Progress tracking for long-running operations

### Important Notes

- **Compatibility**: The project includes `patch_diffusers.py` to handle version compatibility between HuggingFace libraries
- **Performance Target**: Generation should complete within 2 seconds on M4 Max
- **Grid System**: All dimensions are based on 910mm/455mm Japanese architectural grid
- **MVP Focus**: Many components have placeholder implementations; the architecture supports gradual improvement

### Dependencies and Frameworks

- **ML/AI**: PyTorch 2.3.0 (MPS support), Stable Diffusion v1.4, Diffusers 0.19.3, PEFT 0.4.0
- **OCR**: PaddleOCR (primary), EasyOCR (fallback)
- **Graphics**: OpenCV, Pillow, svgwrite
- **CAD**: FreeCAD 0.22 Python API
- **UI**: Streamlit
- **Optimization**: OR-Tools (CP-SAT solver)