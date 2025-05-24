import time
import psutil
import torch # For MPS memory check if applicable
import argparse
import os
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Assuming main_app.py contains a class that can run the full pipeline.
# For now, we will use the placeholder from main_app.py or define a similar one here.
# from ui.main_app import FloorPlanApp # This would be the ideal import

# Using Placeholder from main_app.py for now for CLI execution context
# This requires FloorPlanGeneratorPlaceholder and FreeCADGeneratorPlaceholder
# to be accessible or defined here if main_app.py is not directly runnable as a library.

# Re-defining placeholder here to make script self-contained for now until full integration
class FloorPlanGeneratorPlaceholderForPerfTest:
    def __init__(self):
        print("PerfTest Placeholder: FloorPlanGenerator initialized")
        # self.lora_trainer = LoRATrainer() # Would load trained model here
        # self.constraint_checker = ArchitecturalConstraints()

    def create_site_mask(self, width_grids, height_grids, target_size=(256,256)):
        mask = np.ones(target_size, dtype=np.uint8) * 255 
        pixels_per_grid = target_size[0] / 20 
        mask_width_pixels = int(width_grids * pixels_per_grid)
        mask_height_pixels = int(height_grids * pixels_per_grid)
        start_x = (target_size[0] - mask_width_pixels) // 2
        start_y = (target_size[1] - mask_height_pixels) // 2
        end_x = start_x + mask_width_pixels
        end_y = start_y + mask_height_pixels
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 0, -1) 
        return mask

    def generate_plan(self, site_mask_image, prompt):
        raw_plan_rgba = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        cv2.rectangle(raw_plan_rgba, (30,30), (200,200), (200,0,0,255), 2)
        return raw_plan_rgba

    def validate_constraints(self, raw_plan_image):
        validated_plan_display = raw_plan_image[:,:,:3].copy()
        cv2.putText(validated_plan_display, "Validated", (10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return validated_plan_display

    def to_svg(self, validated_plan_data):
        return "<svg><text>dummy</text></svg>"

class FreeCADGeneratorPlaceholderForPerfTest:
    def create_3d_model(self, validated_plan_data, metadata, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"Dummy FreeCAD model for site {metadata.get('site_grid_size')}")
        return {'fcstd_path': output_path}

# This class simulates the core pipeline logic from FloorPlanApp for performance testing.
class PerformanceTestAppSimulator:
    def __init__(self):
        self.generator = FloorPlanGeneratorPlaceholderForPerfTest()
        self.freecad_gen = FreeCADGeneratorPlaceholderForPerfTest()
        # Ensure output directories are created, similar to main_app.py
        os.makedirs("outputs/generated", exist_ok=True)
        os.makedirs("outputs/svg", exist_ok=True)
        os.makedirs("outputs/freecad", exist_ok=True)

    def full_pipeline(self, width, height, rooms_str="4LDK", style="modern"):
        """Simulates the full generation pipeline based on main_app.py logic."""
        # 1. Create site mask
        site_mask = self.generator.create_site_mask(width, height)
        
        # 2. AI Plan Generation (placeholder)
        prompt = f"site_size_{width}x{height}, rooms_{rooms_str}, style_{style}, japanese_house"
        raw_plan = self.generator.generate_plan(site_mask, prompt)
        
        # 3. Constraint Validation (placeholder)
        validated_plan = self.generator.validate_constraints(raw_plan)
        
        # 4. Vector Conversion (placeholder)
        svg_data = self.generator.to_svg(validated_plan)
        with open(f"outputs/svg/perf_test_{width}x{height}.svg", "w") as f:
            f.write(svg_data)

        # 5. FreeCAD 3D Modeling (placeholder)
        #   The fcstd_generator needs structured data. For placeholder, this is simplified.
        dummy_metadata_for_fc = {'site_grid_size': (width, height)}
        dummy_plan_data_for_fc = {
             'walls': (raw_plan[:,:,0] > 128).astype(int) if raw_plan is not None else np.zeros((256,256), dtype=int)
        } # Simplified input for placeholder

        fc_output_path = f"outputs/freecad/perf_test_model_{width}x{height}.FCStd"
        freecad_result = self.freecad_gen.create_3d_model(
            dummy_plan_data_for_fc, 
            dummy_metadata_for_fc,
            fc_output_path
        )
        return {
            'site_mask': site_mask,
            'raw_plan': raw_plan,
            'validated_plan': validated_plan,
            'svg_path': f"outputs/svg/perf_test_{width}x{height}.svg",
            'fcstd_path': freecad_result['fcstd_path']
        }

def performance_benchmark():
    """システム全体のパフォーマンステスト (Requirement 10.2)"""
    parser = argparse.ArgumentParser(description="Run performance benchmark for floor plan generation.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test case.")
    args = parser.parse_args()

    test_cases = [
        {'name': 'small', 'width': 8, 'height': 8, 'rooms': "3LDK"},
        {'name': 'medium', 'width': 11, 'height': 10, 'rooms': "4LDK"},
        {'name': 'large', 'width': 15, 'height': 12, 'rooms': "5LDK"},
    ]
    
    # app_simulator = PerformanceTestAppSimulator() # Uses placeholder classes
    # When ui.main_app.FloorPlanApp is ready and its generator is the actual one:
    # from ui.main_app import FloorPlanApp # This would eventually be the target
    # app_instance = FloorPlanApp() # This should use the *actual* generator and FreeCAD bridge
    # For now, stick to the local simulator with placeholders for CLI execution.
    app_simulator = PerformanceTestAppSimulator()
    
    results_summary = []
    all_run_details = []

    for case in test_cases:
        print(f"\nTesting case: {case['name']} ({case['width']}x{case['height']} grid)... For {args.runs} runs.")
        case_times = []
        case_memory_usages = []
        case_gpu_usages = [] # MPS memory

        for i in range(args.runs):
            print(f"  Run {i+1}/{args.runs}...")
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            initial_gpu_mem = 0
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                initial_gpu_mem = torch.mps.current_allocated_memory() / (1024 * 1024) # MB
            
            start_time = time.time()
            
            try:
                # This should call the main generation pipeline
                # result = app_instance.generator.full_pipeline(case['width'], case['height'], case['rooms'], "modern")
                # Using the simulator for now:
                result = app_simulator.full_pipeline(case['width'], case['height'], case['rooms'])
                
                end_time = time.time()
                processing_time = end_time - start_time
                case_times.append(processing_time)
                
                final_memory = process.memory_info().rss / (1024 * 1024)
                memory_usage = final_memory - initial_memory
                case_memory_usages.append(memory_usage)

                gpu_usage = 0
                if torch.backends.mps.is_available():
                    final_gpu_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
                    gpu_usage = final_gpu_mem - initial_gpu_mem
                case_gpu_usages.append(gpu_usage)

                run_detail = {
                    'case': case['name'],
                    'run': i + 1,
                    'processing_time_s': processing_time,
                    'memory_usage_mb': memory_usage,
                    'gpu_usage_mb': gpu_usage,
                    'success': True
                }
                all_run_details.append(run_detail)
                print(f"    Run {i+1} Success: {processing_time:.2f}s, RAM: {memory_usage:.1f}MB, GPU: {gpu_usage:.1f}MB")

            except Exception as e:
                print(f"    Run {i+1} FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                all_run_details.append({
                    'case': case['name'], 'run': i + 1, 'error': str(e), 'success': False
                })
        
        if case_times: # If any runs were successful
            avg_time = np.mean(case_times)
            avg_mem = np.mean(case_memory_usages)
            avg_gpu = np.mean(case_gpu_usages)
            results_summary.append({
                'case_name': case['name'],
                'avg_processing_time_s': avg_time,
                'avg_memory_usage_mb': avg_mem,
                'avg_gpu_usage_mb': avg_gpu,
                'num_successful_runs': len(case_times),
                'total_runs': args.runs,
                'time_target_s': 5.0,
                'time_target_achieved': avg_time <= 5.0 if case_times else False
            })

    print("\n📊 Performance Benchmark Summary:")
    print("-------------------------------------------------------------------------------------")
    print(f"| {'Test Case':<10} | {'Avg Time(s)':<12} | {'Avg RAM (MB)':<14} | {'Avg GPU (MB)':<14} | {'Success':<10} |")
    print("|--------------|----------------|------------------|------------------|------------|")
    total_successful_runs = 0
    total_target_achieved = 0
    for res in results_summary:
        print(f"| {res['case_name']:<12} | {res['avg_processing_time_s']:<14.2f} | {res['avg_memory_usage_mb']:<16.1f} | {res['avg_gpu_usage_mb']:<16.1f} | {str(res['time_target_achieved']):<10} |")
        total_successful_runs += res['num_successful_runs']
        if res['time_target_achieved']:
            total_target_achieved +=1 
            
    print("-------------------------------------------------------------------------------------")
    
    num_test_cases = len(test_cases)
    overall_time_compliance = (total_target_achieved / num_test_cases * 100) if num_test_cases > 0 else 0
    print(f"Overall time target compliance ({total_target_achieved}/{num_test_cases} cases): {overall_time_compliance:.1f}%")
    
    # print("\nFull Run Details:", all_run_details) # Optional: print all details
    return results_summary, all_run_details

if __name__ == "__main__":
    # Need numpy and cv2 for placeholder to run.
    # These should be in requirements.txt if this script is to be run directly with placeholders.
    import numpy as np
    import cv2
    performance_benchmark() 