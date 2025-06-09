import argparse
import os
import sys
import numpy as np
import cv2 # For saving image

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# These would be the actual implementation classes
# from inference.generator import FloorPlanGenerator # Assuming this will be the main generator class
# from freecad_bridge.fcstd_generator import FreeCADGenerator

from training.dataset import FloorPlanDataset

# Using Placeholders for now as the actual classes might not be fully integrated or might depend on UI state
class FloorPlanGeneratorPlaceholderCLI:
    def __init__(self, model_path="models/lora_weights/epoch_15"): # Default to a late epoch
        print(f"CLI Placeholder: FloorPlanGenerator initialized (model: {model_path})")
        # In a real scenario, load the LoRA model here
        # self.trainer = LoRATrainer()
        # self.trainer.load_lora_weights(model_path) # Assuming LoRATrainer has load method
        # self.constraint_checker = ArchitecturalConstraints()

    def create_site_mask(self, width_grids, height_grids, target_size=(256,256)):
        print(f"CLI Placeholder: Creating site mask for {width_grids}x{height_grids} grids")
        mask = np.ones(target_size, dtype=np.uint8) * 255
        pixels_per_grid = target_size[0] / 20 # Consistent with TrainingDataGenerator
        mask_width_pixels = int(width_grids * pixels_per_grid)
        mask_height_pixels = int(height_grids * pixels_per_grid)
        start_x = (target_size[0] - mask_width_pixels) // 2
        start_y = (target_size[1] - mask_height_pixels) // 2
        end_x = start_x + mask_width_pixels
        end_y = start_y + mask_height_pixels
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 0, -1) # Black site area on white bg
        return mask

    def generate_plan(self, site_mask_image, prompt):
        print(f"CLI Placeholder: Generating plan with prompt: {prompt}")
        # Dummy raw plan (RGBA HWC format)
        raw_plan_rgba = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        # Simulate some structure
        cv2.rectangle(raw_plan_rgba, (30,30), (200,200), (200,0,0,255), 2) # Walls in Red-ish channel (simulated)
        raw_plan_rgba[50:70, :, 0] = 220 # Example Wall
        raw_plan_rgba[:, 80:100, 0] = 220 # Example Wall
        raw_plan_rgba[55:65, 55:100, 1] = 200 # Example Opening in Green channel
        raw_plan_rgba[100:120, 100:130, 2] = 200 # Example Stairs in Blue channel
        return raw_plan_rgba

    def validate_constraints(self, raw_plan_image):
        print("CLI Placeholder: Validating constraints")
        # Return a displayable image (BGR) for now
        validated_plan_display = raw_plan_image[:,:,:3].copy() # Take RGB
        cv2.putText(validated_plan_display, "Validated", (10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return validated_plan_display # BGR HWC

    def to_svg(self, validated_plan_data):
        print("CLI Placeholder: Converting to SVG")
        return "<svg width=\"100\" height=\"100\"><rect width=\"100\" height=\"100\" style=\"fill:rgb(200,200,200);stroke-width:3;stroke:rgb(0,0,0)\" /><text x=\"10\" y=\"50\" fill=\"black\">Dummy SVG</text></svg>"

class FreeCADGeneratorPlaceholderCLI:
    def create_3d_model(self, validated_plan_data, metadata, output_path):
        print(f"CLI Placeholder: Creating 3D model at {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"Dummy FreeCAD model for site {metadata.get('site_grid_size')}")
        return {'fcstd_path': output_path}

def create_integrated_metadata(args):
    """Create integrated JSON-style metadata from CLI arguments"""
    room_count = 3
    if args.rooms and args.rooms[0].isdigit():
        room_count = int(args.rooms[0])
    
    structural_elements = []
    
    structural_elements.append({
        "type": "stair",
        "grid_x": args.stair_x,
        "grid_y": args.stair_y,
        "grid_width": 1.6,
        "grid_height": 0.7,
        "name": "stair_1"
    })
    
    if args.entrance_x is not None and args.entrance_y is not None and args.current_floor == "1F":
        structural_elements.append({
            "type": "entrance",
            "grid_x": args.entrance_x,
            "grid_y": args.entrance_y,
            "grid_width": 1.3,
            "grid_height": 1.1,
            "name": "entrance_1"
        })
    
    total_grids = args.width * args.height
    if args.current_floor == "1F":
        zones = [
            {"type": "living", "approximate_grids": int(total_grids * 0.4), "priority": 1},
            {"type": "service", "approximate_grids": int(total_grids * 0.3), "priority": 2},
            {"type": "circulation", "approximate_grids": int(total_grids * 0.3), "priority": 3}
        ]
    else:
        zones = [
            {"type": "private", "approximate_grids": int(total_grids * 0.6), "priority": 1},
            {"type": "work_space", "approximate_grids": int(total_grids * 0.2), "priority": 2},
            {"type": "circulation", "approximate_grids": int(total_grids * 0.2), "priority": 3}
        ]
    
    metadata = {
        "grid_dimensions": {
            "width_grids": args.width,
            "height_grids": args.height
        },
        "scale_info": {
            "drawing_scale": "1:100",
            "grid_mm": 910
        },
        "building_context": {
            "type": args.building_type,
            "floors_total": args.floors,
            "current_floor": args.current_floor,
            "typical_patterns": {
                "1F": ["entrance_area", "stair", "public_living_space", "wet_areas", "storage_zones"],
                "2F": ["stair", "private_sleeping_areas", "work_space", "utility_area", "balcony"]
            }
        },
        "structural_elements": structural_elements,
        "zones": zones,
        "training_hints": {
            "total_area_grids": total_grids,
            "room_count": room_count,
            "has_entrance": args.current_floor == "1F" and args.entrance_x is not None,
            "has_stair": True,
            "has_balcony": args.current_floor != "1F",
            "floor_constraints": {
                "required_elements": ["stair"],
                "prohibited_elements": ["entrance"] if args.current_floor != "1F" else []
            }
        }
    }
    
    return metadata

def generate_enhanced_prompt(args):
    """Generate enhanced prompt using integrated metadata format"""
    metadata = create_integrated_metadata(args)
    
    temp_dataset = FloorPlanDataset(data_dir="dummy")
    
    enhanced_prompt = temp_dataset.generate_integrated_prompt(metadata)
    
    return enhanced_prompt

def main():
    parser = argparse.ArgumentParser(description="Generate a floor plan using the trained model.")
    parser.add_argument("--width", type=int, default=11, help="Width of the site in grid units.")
    parser.add_argument("--height", type=int, default=10, help="Height (depth) of the site in grid units.")
    parser.add_argument("--rooms", type=str, default="4LDK", help="Number of rooms (e.g., 3LDK, 4LDK). Used in prompt.")
    parser.add_argument("--style", type=str, default="modern", help="Architectural style (e.g., modern, traditional).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated files.")
    parser.add_argument("--model_path", type=str, default="models/lora_weights/epoch_15", 
                        help="Path to the trained LoRA model weights directory.")
    parser.add_argument("--no_3d", action="store_true", help="Skip FreeCAD 3D model generation.")
    parser.add_argument("--floors", type=int, default=2, help="Total number of floors in the building.")
    parser.add_argument("--current_floor", type=str, default="1F", help="Current floor being generated (e.g., 1F, 2F).")
    parser.add_argument("--building_type", type=str, default="single_family_house", help="Type of building (e.g., single_family_house, apartment).")
    parser.add_argument("--enhanced_prompt", action="store_true", help="Use enhanced prompt generation matching training dataset format.")
    parser.add_argument("--stair_x", type=float, default=0.5, help="Stair X position in grid coordinates (for enhanced prompts).")
    parser.add_argument("--stair_y", type=float, default=3.0, help="Stair Y position in grid coordinates (for enhanced prompts).")
    parser.add_argument("--entrance_x", type=float, default=None, help="Entrance X position in grid coordinates (optional, for enhanced prompts).")
    parser.add_argument("--entrance_y", type=float, default=None, help="Entrance Y position in grid coordinates (optional, for enhanced prompts).")

    args = parser.parse_args()

    # Create output subdirectories if they don't exist
    generated_img_dir = os.path.join(args.output_dir, "generated")
    svg_dir = os.path.join(args.output_dir, "svg")
    fc_dir = os.path.join(args.output_dir, "freecad")
    os.makedirs(generated_img_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    if not args.no_3d:
        os.makedirs(fc_dir, exist_ok=True)

    # Initialize generators (using placeholders for CLI)
    plan_generator = FloorPlanGeneratorPlaceholderCLI(model_path=args.model_path)
    if not args.no_3d:
        freecad_generator = FreeCADGeneratorPlaceholderCLI()

    print(f"Generating floor plan for a {args.width}x{args.height} site, rooms: {args.rooms}, style: {args.style}")

    try:
        # 1. Create site mask
        site_mask = plan_generator.create_site_mask(args.width, args.height)
        cv2.imwrite(os.path.join(generated_img_dir, "site_mask_cli.png"), site_mask)
        print(f"Site mask saved to {generated_img_dir}/site_mask_cli.png")

        # 2. Generate plan using AI (placeholder)
        if args.enhanced_prompt:
            prompt = generate_enhanced_prompt(args)
            print(f"Using enhanced prompt generation")
        else:
            prompt = f"site_size_{args.width}x{args.height}, rooms_{args.rooms}, style_{args.style}, japanese_house, 910mm_grid, architectural_plan"
            print(f"Using legacy prompt generation")
        # raw_plan is HWC RGBA from placeholder
        raw_plan_rgba = plan_generator.generate_plan(site_mask, prompt)
        cv2.imwrite(os.path.join(generated_img_dir, "plan_raw_cli.png"), cv2.cvtColor(raw_plan_rgba, cv2.COLOR_RGBA2BGRA))
        print(f"Raw plan saved to {generated_img_dir}/plan_raw_cli.png")

        # 3. Validate constraints (placeholder)
        # validate_constraints placeholder returns BGR HWC
        validated_plan_bgr = plan_generator.validate_constraints(raw_plan_rgba) 
        cv2.imwrite(os.path.join(generated_img_dir, "plan_validated_cli.png"), validated_plan_bgr)
        print(f"Validated plan saved to {generated_img_dir}/plan_validated_cli.png")

        # 4. Convert to SVG (placeholder)
        # to_svg takes the validated plan data (which could be grid data or image)
        svg_data = plan_generator.to_svg(validated_plan_bgr) 
        svg_path = os.path.join(svg_dir, "plan_cli.svg")
        with open(svg_path, 'w') as f:
            f.write(svg_data)
        print(f"SVG saved to {svg_path}")

        # 5. Generate FreeCAD 3D model (placeholder)
        if not args.no_3d:
            print("Generating FreeCAD model...")
            # The FreeCAD generator needs structured data, not just an image.
            # The placeholder `dummy_validated_plan_grids_dict` from main_app.py is an example.
            # We need to ensure `validate_constraints` (or its real counterpart) provides this.
            # For CLI placeholder, we pass a simplified metadata.
            dummy_metadata = {
                'site_grid_size': (args.width, args.height)
            }
            # The fcstd_generator needs input based on its `create_3d_model` signature.
            # For placeholder, it's just a dummy operation.
            fc_output_path = os.path.join(fc_dir, f"model_cli_{args.width}x{args.height}.FCStd")
            freecad_result = freecad_generator.create_3d_model(validated_plan_bgr, dummy_metadata, fc_output_path)
            print(f"FreeCAD model generation (placeholder) complete: {freecad_result['fcstd_path']}")

        print("\nGeneration process (using placeholders) complete.")
        print(f"Outputs are in the '{args.output_dir}' directory.")

    except Exception as e:
        print(f"An error occurred during plan generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  