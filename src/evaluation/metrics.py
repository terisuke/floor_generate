import numpy as np
import os
import json
# Add src directory to Python path to allow module imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "..")) # Go up two levels to project root, then src
if src_dir not in sys.path:
    sys.path.insert(0, os.path.join(src_dir, "src"))

# Assuming architectural_constraints.py is in src/constraints
from constraints.architectural_constraints import ArchitecturalConstraints

class QualityMetrics:
    def __init__(self):
        self.constraints_checker = ArchitecturalConstraints() # For constraint satisfaction

    def check_dimension_accuracy(self, plan_image_data, metadata):
        """生成されたプランの寸法精度を評価 (Placeholder)
           plan_image_data: 生成されたプランのグリッドデータまたは画像データ
           metadata: 元のPDFの寸法情報や敷地情報を含むメタデータ
           戻り値: 寸法精度スコア (例: 0.0 - 1.0)
        """
        print("Checking dimension accuracy (placeholder)...")
        # This is complex. It would involve:
        # 1. Detecting dimensions from the *generated* plan_image_data (similar to DimensionExtractor on source PDFs).
        # 2. Comparing these detected dimensions against the normalized dimensions in `metadata` or target dimensions.
        # 3. Calculating an error metric (e.g., average percentage error, number of matching dimensions).
        # For MVP, this is a placeholder returning a dummy score.
        original_dims = metadata.get('normalized_dimensions', [])
        if not original_dims:
            return 0.5 # Cannot compare if no original dimensions
        
        # Dummy: Assume 80% accuracy for now if original dimensions exist.
        return 0.8 

    def calculate_room_areas(self, plan_image_data_or_grid):
        """プランから部屋ごとの面積を計算 (Placeholder)
           plan_image_data_or_grid: 部屋情報を含むグリッドデータ (例: CP-SATの出力)
                                      または部屋チャンネルを含む画像。
           戻り値: 部屋IDごとの面積(グリッド数)の辞書 {room_id: area_grids}
        """
        print("Calculating room areas (placeholder)...")
        # This requires identifying distinct rooms and counting their grid cells.
        # If `plan_image_data_or_grid` is the output from CP-SAT (`solution['rooms']`),
        # it's a grid where each cell has a room ID.
        # Example: plan_image_data_or_grid = solution['rooms'] from ArchitecturalConstraints
        
        room_areas = {}
        if isinstance(plan_image_data_or_grid, np.ndarray) and plan_image_data_or_grid.ndim == 2:
            unique_room_ids = np.unique(plan_image_data_or_grid)
            for room_id in unique_room_ids:
                if room_id == 0: continue # Skip non-room cells
                area = np.sum(plan_image_data_or_grid == room_id)
                room_areas[int(room_id)] = int(area)
        else:
            # Fallback if input is not the expected room grid
            room_areas = {1: 10, 2: 12, 3: 9} # Dummy areas
        
        print(f"Calculated room areas (grid cells): {room_areas}")
        return room_areas

    def validate_room_areas(self, room_areas_dict):
        """部屋面積の妥当性を検証 (Placeholder)
           room_areas_dict: {room_id: area_grids} 
           戻り値: 妥当性スコア (例: 0.0 - 1.0, 1.0は全ての部屋が妥当)
        """
        print("Validating room areas (placeholder)...")
        if not room_areas_dict: return 0.0
        
        # Example: Check if areas are within a reasonable range (e.g., 6sqm to 30sqm)
        # 1 grid cell is approx 0.91*0.91 = 0.8281 sqm
        min_grids = int(np.ceil(6 / 0.8281))  # ~8 grids
        max_grids = int(np.floor(30 / 0.8281)) # ~36 grids
        
        valid_rooms = 0
        for room_id, area_grids in room_areas_dict.items():
            if min_grids <= area_grids <= max_grids:
                valid_rooms += 1
        
        score = valid_rooms / len(room_areas_dict) if len(room_areas_dict) > 0 else 0
        print(f"Room area validation score: {score:.2f}")
        return score

    def evaluate_circulation(self, plan_image_data_or_grid):
        """動線の妥当性を評価 (Placeholder)
           plan_image_data_or_grid: 部屋、壁、開口部情報を含むデータ
           戻り値: 動線スコア (例: 0.0 - 1.0)
        """
        print("Evaluating circulation (placeholder)...")
        # Complex: requires pathfinding between rooms, entrance to rooms, etc.
        # Check for accessibility, dead ends, narrow corridors.
        # For MVP, return a dummy score.
        return 0.7 # Dummy score

    def calculate_overall_score(self, metrics_dict):
        """複数のメトリクスから総合スコアを計算 (Placeholder)"""
        # Simple weighted average or minimum of scores
        weights = {
            'constraint_satisfaction': 0.4,
            'dimension_accuracy': 0.2,
            'valid_room_areas': 0.2,
            'circulation_score': 0.2
        }
        overall_score = 0
        if metrics_dict.get('constraint_satisfaction'): # Boolean to float
            overall_score += float(metrics_dict['constraint_satisfaction']) * weights['constraint_satisfaction']
        overall_score += metrics_dict.get('dimension_accuracy', 0) * weights['dimension_accuracy']
        overall_score += metrics_dict.get('valid_room_areas', 0) * weights['valid_room_areas']
        overall_score += metrics_dict.get('circulation_score', 0) * weights['circulation_score']
        print(f"Calculated overall score: {overall_score:.2f}")
        return overall_score

    def evaluate_generated_plan(self, plan_image_data, metadata):
        """生成プランの品質評価 (Requirement 10.1)
           plan_image_data: AIによって生成されたプランの画像データ(例: RGBA numpy array)
                          またはCP-SATソルバーへの入力となるグリッドデータ。
                          `ArchitecturalConstraints.image_to_grid` が処理できる形式を想定。
           metadata: 元のPDFの寸法情報や敷地情報を含むメタデータ
        """
        print(f"Evaluating generated plan (source: {metadata.get('source_pdf', 'unknown')})...")
        metrics = {}

        # 1. 制約充足率
        # The `validate_and_fix` method of ArchitecturalConstraints takes an image,
        # converts it to a grid, applies constraints, and returns a solution (or None).
        # We use its success as an indicator of constraint satisfaction.
        # The input `plan_image_data` should be what `image_to_grid` expects.
        # Typically, this would be the raw AI output *before* CP-SAT fixing.
        # If CP-SAT fixing is part of the generation pipeline already, then this metric
        # might measure if the *fixed* plan is valid (which it should be if solver succeeded).
        # Let's assume plan_image_data is the AI output to be validated.
        try:
            constraint_solution = self.constraints_checker.validate_and_fix(plan_image_data)
            metrics['constraint_satisfaction'] = constraint_solution is not None
            # If a solution is found, we can use this `constraint_solution` (dict of grids)
            # for further metric calculations (room areas, etc.)
            plan_data_for_metrics = constraint_solution if constraint_solution is not None else None
        except Exception as e:
            print(f"Error during constraint checking: {e}")
            metrics['constraint_satisfaction'] = False
            plan_data_for_metrics = None # Cant use for further metrics

        # 2. 寸法精度
        # Needs the generated plan image/data and original metadata
        metrics['dimension_accuracy'] = self.check_dimension_accuracy(plan_image_data, metadata)

        # 3. 部屋面積妥当性
        # Requires room segmentation from the generated (and possibly fixed) plan.
        # If constraint_solution is available and contains room segmentation, use it.
        if plan_data_for_metrics and 'rooms' in plan_data_for_metrics:
            room_areas = self.calculate_room_areas(plan_data_for_metrics['rooms'])
            metrics['valid_room_areas'] = self.validate_room_areas(room_areas)
        else:
            # Fallback if no room grid data from CP-SAT (e.g. it failed, or we use raw image)
            # This would require running room detection on `plan_image_data`
            # Placeholder: assume rooms can be roughly detected from a channel of plan_image_data
            if plan_image_data.ndim ==3 and plan_image_data.shape[2] == 4: # RGBA
                 room_channel_dummy = (plan_image_data[:,:,3] > 128).astype(int) # Alpha as rooms
                 room_areas_from_img = self.calculate_room_areas(room_channel_dummy) 
                 metrics['valid_room_areas'] = self.validate_room_areas(room_areas_from_img)
            else:
                metrics['valid_room_areas'] = 0.3 # Low dummy score

        # 4. 動線妥当性
        # Requires wall, room, and opening information from the generated/fixed plan.
        # If plan_data_for_metrics is available, it can be used.
        metrics['circulation_score'] = self.evaluate_circulation(plan_data_for_metrics or plan_image_data)

        # 5. 総合スコア
        metrics['overall_score'] = self.calculate_overall_score(metrics)
        
        print(f"Metrics for {metadata.get('source_pdf', 'unknown')}: {metrics}")
        return metrics

    def batch_evaluation(self, test_cases_dir):
        """バッチ評価実行 (Requirement 10.1)
           test_cases_dir: 学習/検証用データペア (pair_xxxx) が格納されたディレクトリ。
                           各ペアには target plan image (e.g., floor_plan.png) と metadata.json が含まれる。
                           ここでは、AIが生成したプランを評価することを想定。この関数はテストケースの
                           *ground truth* を読み込み、それに対して (シミュレートされた)AI生成プランを評価する形になる。
                           または、事前に生成されたAIプランのリストを読み込む。
                           要件定義書の`performance_test.py`内の`app.generator.full_pipeline`がAIプランを生成する。
                           このメソッドは、その生成結果と対応するメタデータを評価する。
        """
        print(f"Starting batch evaluation on data from: {test_cases_dir}")
        results = []
        success_count = 0
        
        # This needs a list of (generated_plan_image, metadata) to evaluate.
        # For now, let's assume we iterate through prepared test data pairs (ground truth for now)
        # and simulate a "generated" plan for each to test the metrics calculation.

        pair_dirs = glob(f"{test_cases_dir}/pair_*")
        if not pair_dirs:
            print(f"No test cases (pair_xxxx dirs) found in {test_cases_dir}")
            return [], {}

        num_cases_to_eval = min(len(pair_dirs), 10) # Limit to 10 cases for quick test
        print(f"Evaluating up to {num_cases_to_eval} test cases...")

        for i, pair_dir in enumerate(pair_dirs[:num_cases_to_eval]):
            case_id = os.path.basename(pair_dir)
            print(f"\nProcessing test case: {case_id}")
            try:
                metadata_path = os.path.join(pair_dir, "metadata.json")
                floor_plan_path = os.path.join(pair_dir, "floor_plan.png") # This is ground truth plan

                if not (os.path.exists(metadata_path) and os.path.exists(floor_plan_path)):
                    print(f"Skipping {case_id}: missing metadata or floor plan image.")
                    results.append({'case_id': case_id, 'error': 'Missing files', 'success': False})
                    continue

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Simulate an AI-generated plan. For testing metrics, this might be:
                # 1. The ground truth itself (to see if metrics yield high scores).
                # 2. A deliberately flawed version of the ground truth.
                # 3. Actual output from the AI model if available.
                # For now, use the ground truth image as a stand-in for "generated plan".
                # This floor_plan.png is RGBA HWC uint8 from TrainingDataGenerator
                generated_plan_image_rgba = cv2.imread(floor_plan_path, cv2.IMREAD_UNCHANGED)
                if generated_plan_image_rgba is None:
                    print(f"Skipping {case_id}: could not load floor plan image {floor_plan_path}")
                    results.append({'case_id': case_id, 'error': 'Image load failed', 'success': False})
                    continue

                # Convert to format expected by evaluate_generated_plan (e.g., normalized [0,1] float)
                # If image_to_grid expects [0,1] float, then normalize.
                # Current image_to_grid in ArchitecturalConstraints assumes >0.5 for walls if float.
                # Let's pass the RGBA numpy array [0-255] directly, image_to_grid handles it.
                # For consistency, let's assume evaluate_generated_plan expects the same format
                # as the AI output (e.g. normalized float32 HWC RGBA)
                generated_plan_float32 = generated_plan_image_rgba.astype(np.float32) / 255.0

                metrics = self.evaluate_generated_plan(generated_plan_float32, metadata)
                
                is_success = metrics['overall_score'] > 0.6 # Target success threshold
                results.append({
                    'case_id': case_id,
                    'metrics': metrics,
                    'success': is_success
                })
                
                if is_success:
                    success_count += 1
                    
            except Exception as e:
                print(f"Error evaluating {case_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'case_id': case_id,
                    'error': str(e),
                    'success': False
                })
        
        total_evaluated = len(pair_dirs[:num_cases_to_eval])
        success_rate = success_count / total_evaluated if total_evaluated > 0 else 0
        report = {
            'total_cases_attempted': total_evaluated,
            'successful_evaluations': success_count,
            'success_rate': success_rate,
            'target_rate': 0.6,  # Target 60% quality
            'achieved_target': success_rate >= 0.6
        }
        
        print(f"\nBatch evaluation report: {report}")
        return results, report

# Example Usage:
# if __name__ == '__main__':
#     metrics_evaluator = QualityMetrics()
    
#     # Create dummy metadata and plan data for a single evaluation test
#     dummy_metadata = {
#         'source_pdf': 'dummy_test.pdf',
#         'normalized_dimensions': [{'type': 'linear', 'value': 10000, 'grid_type': 'primary'}],
#         'site_grid_size': (11,10), # (width, height in grids)
#         'total_area_sqm': 11*0.91*10*0.91,
#         'room_count': 4
#     }
#     # Dummy AI output (RGBA, HWC, float32, 0-1 range)
#     dummy_plan_data = np.random.rand(256, 256, 4).astype(np.float32)
#     dummy_plan_data[:,:,0] = (np.random.rand(256,256) > 0.8).astype(np.float32) # Simulate some walls
#     dummy_plan_data[:,:,3] = (np.random.rand(256,256) > 0.5).astype(np.float32) # Simulate some rooms in alpha

#     single_metrics = metrics_evaluator.evaluate_generated_plan(dummy_plan_data, dummy_metadata)
#     print("\nSingle Plan Evaluation Metrics:")
#     for key, value in single_metrics.items():
#         print(f"  {key}: {value}")

#     # For batch evaluation, point to a directory with `pair_xxxx` subfolders
#     # Ensure that these folders contain metadata.json and floor_plan.png (ground truth RGBA image)
#     # dummy_test_cases_dir = "data/training" # Or "data/validation"
#     # if os.path.exists(dummy_test_cases_dir) and os.listdir(dummy_test_cases_dir):
#     #     batch_results, batch_report = metrics_evaluator.batch_evaluation(dummy_test_cases_dir)
#     #     # print("\nBatch Evaluation Full Results:", batch_results)
#     #     print("\nBatch Evaluation Summary Report:", batch_report)
#     # else:
#     #     print(f"Skipping batch evaluation: Directory {dummy_test_cases_dir} is empty or does not exist.") 