# AGENTS.md - 910mmグリッド住宅プラン自動生成システム

Welcome to the 910mm Grid Housing Plan Generation System. This file contains guidelines for contributors and AI assistants working on this architectural AI system.

## Repository overview
- **Source code**: `src/` contains preprocessing, training, inference, constraints, freecad_bridge, and ui modules.
- **Data pipeline**: `data/` with raw_pdfs, extracted, normalized, training, and validation directories.
- **Models**: `models/` for trained LoRA weights and checkpoints.
- **Scripts**: `scripts/` for setup, training, generation, and performance testing.
- **Outputs**: `outputs/` for generated plans in PNG, SVG, DXF, and FreeCAD formats.
- **Tests**: `tests/` for unit, integration, and system testing.

## Local workflow
1. Set up environment and install dependencies:
   ```bash
   # macOS setup
   cd ~/repos/floor_generate
   brew install python@3.11 freecad git cmake pkg-config poppler tesseract tesseract-lang
   python3.11 -m venv floorplan_env
   source floorplan_env/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install torch==2.3.0 torchvision torchaudio  # MPS support
   pip install -r requirements.txt
   chmod +x setup_dirs.sh && ./setup_dirs.sh
   
   # Ubuntu 22.04 setup
   sudo apt install -y python3.11 python3.11-venv python3.11-dev poppler-utils tesseract-ocr tesseract-ocr-jpn
   python3.11 -m venv floorplan_env
   source floorplan_env/bin/activate
   pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

2. Format, lint and type-check your changes:
   ```bash
   black .                   # Format code
   isort .                   # Sort imports
   flake8 .                  # Lint code
   mypy src/                 # Type checking
   ```

3. Run the tests:
   ```bash
   pytest                    # Run all tests
   pytest --cov=src         # With coverage
   pytest tests/unit/       # Unit tests only
   pytest tests/integration/ # Integration tests
   ```

4. Run the application pipeline:
   ```bash
   # Data preprocessing
   python scripts/process_pdfs.py
   python scripts/prepare_training_data.py --pdf_dir data/raw_pdfs --output_dir data/training
   
   # Model training
   python scripts/train_model.py --data_dir data/training --epochs 20
   
   # Generation
   python scripts/generate_plan.py --width 11 --height 10 --output outputs/
   
   # Streamlit UI
   streamlit run src/ui/main_app.py --server.port 8501
   ```

## Testing guidelines
Maintain high test coverage for architectural constraints and AI pipeline:
```bash
pytest -v                   # Verbose output with detailed test results
pytest -x                   # Stop on first failure for debugging
pytest --cov=src --cov-report=html  # Generate HTML coverage report
python scripts/performance_test.py  # System performance benchmark (5s target)
```
- Test all PDF extraction and dimension normalization algorithms
- Validate AI-generated plans against architectural constraints
- Test FreeCAD integration and 3D model generation
- Include edge cases for grid normalization (910mm/455mm mixed grid)
- Performance test: ensure generation completes within 5 seconds per plan

## Style notes
- Follow PEP 8 and use Black formatter with 88-character line limit.
- Use type hints for all function parameters and return values, especially for CAD coordinate data.
- Write comprehensive docstrings for AI model training and architectural constraint functions.
- Use descriptive variable names for architectural elements: `wall_thickness`, `grid_size_mm`, `room_area_sqm`.
- Document units clearly in variable names and comments (mm, sqm, grid_count).

## Commit message format
Use conventional commit format with architectural/AI-specific scopes:
```
type(scope): description

Examples:
feat(pdf-extraction): implement OCR-based dimension detection from architectural drawings
fix(grid-normalizer): resolve 910mm/455mm mixed grid alignment issues
refactor(ai-training): optimize LoRA training memory usage for M4 Max
test(constraints): add CP-SAT validation for stair placement rules
docs(freecad): update 3D model generation workflow documentation
perf(inference): reduce generation time from 8s to 4s per plan
```

## Pull request expectations
PRs should include:
- **Summary**: Clear description of architectural/AI functionality changes
- **Visual validation**: Screenshots of generated floor plans or 3D models
- **Performance impact**: Generation time and memory usage measurements
- **Constraint validation**: Confirmation that architectural rules are maintained
- **CAD compatibility**: Verification that FreeCAD models open and edit properly

Before submitting, ensure:
- [ ] All tests pass (`pytest`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Code is formatted (`black .` and `isort .`)
- [ ] Generated plans pass architectural constraint validation
- [ ] Performance target met (5 seconds per generation)
- [ ] FreeCAD integration produces editable 3D models
- [ ] GPU/MPS memory usage is optimized

## What reviewers look for
- **AI model quality**: Generated plans meet architectural standards and constraints.
- **Grid precision**: 910mm/455mm grid alignment with <5% dimensional error.
- **CAD integration**: FreeCAD models maintain wall thickness, room connectivity.
- **Performance**: Generation pipeline meets 5-second target consistently.
- **Data pipeline**: PDF extraction accuracy and training data quality.
- **Constraint satisfaction**: CP-SAT validation ensures buildable designs.

## Architecture guidelines
- Use modular pipeline: PDF extraction → Grid normalization → AI training → Inference → Constraint validation → CAD generation.
- Implement separate concerns: preprocessing, training, inference, constraints, CAD bridge.
- Apply Clean Architecture principles for AI model training and inference.
- Use dependency injection for model loading and CAD system integration.
- Maintain clear data flow between pipeline stages with well-defined interfaces.

## AI/ML best practices (Stable Diffusion + LoRA)
- Use LoRA (Low-Rank Adaptation) for efficient fine-tuning on architectural data.
- Implement proper data augmentation for floor plan training data.
- Monitor training loss and validation metrics for architectural constraint compliance.
- Use mixed precision training for memory efficiency on M4 Max GPU.
- Apply proper regularization to prevent overfitting on limited architectural datasets.
- Cache preprocessed training data to reduce I/O overhead during training.

## Architectural constraint system (CP-SAT)
- Define hard constraints: wall connectivity, room accessibility, stair placement.
- Implement soft constraints: room size preferences, circulation efficiency.
- Use constraint satisfaction to validate and repair AI-generated plans.
- Apply minimal modification principle: preserve AI design intent while ensuring buildability.
- Test constraint solver performance on various plan sizes and complexity.

## FreeCAD integration best practices
- Use FreeCAD Python API for programmatic 3D model generation.
- Maintain parametric editability: wall heights, thicknesses, room dimensions.
- Generate proper architectural elements: walls, openings, stairs, floors.
- Export compatibility: ensure DXF, STEP, and IGES formats work correctly.
- Implement error handling for CAD geometry generation edge cases.

## Performance optimization guidelines
- Optimize PyTorch memory usage for training and inference on Apple Silicon.
- Use efficient image processing for PDF extraction and grid conversion.
- Cache intermediate results in the data preprocessing pipeline.
- Implement batch processing for multiple plan generation.
- Monitor and optimize constraint solver performance for large plans.
- Profile critical paths: PDF processing, AI inference, CAD generation.

## Data management for architectural AI
- Maintain consistent units: millimeters for dimensions, square meters for areas.
- Implement robust PDF parsing for various architectural drawing formats.
- Validate extracted dimensions against known architectural standards.
- Handle missing or corrupted data in PDF extraction gracefully.
- Store training data with proper metadata: building type, size, style.
- Implement version control for trained models and architectural constraint rules.

## Security and validation
- Validate all input dimensions against reasonable architectural bounds.
- Sanitize user inputs for grid sizes and generation parameters.
- Implement bounds checking for AI-generated coordinates and dimensions.
- Validate generated plans against building codes and safety requirements.
- Handle potentially malicious PDF files safely during processing.
- Ensure FreeCAD file generation doesn't execute arbitrary code.

## Specific technology considerations

### PyTorch/Diffusers (Apple Silicon)
- Use MPS backend for GPU acceleration on M4 Max.
- Implement proper memory management for large model training.
- Handle float16 precision issues on Apple Silicon appropriately.
- Monitor thermal throttling during intensive training sessions.

### OCR and PDF Processing
- Configure EasyOCR for Japanese architectural text recognition.
- Handle various PDF formats and drawing scales consistently.
- Implement robust dimension extraction from architectural symbols.
- Validate OCR results against expected architectural dimension ranges.

### FreeCAD Python API
- Handle FreeCAD document lifecycle properly (create, modify, save, close).
- Implement proper error handling for geometry creation failures.
- Maintain compatibility across different FreeCAD versions.
- Test 3D model generation with various architectural configurations.

### Performance monitoring
- Track generation pipeline performance metrics continuously.
- Monitor memory usage patterns during training and inference.
- Implement automated performance regression testing.
- Profile constraint solver performance on complex architectural layouts.
