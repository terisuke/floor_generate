import streamlit as st
import numpy as np
import os
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inference.generator import FloorPlanGenerator
from freecad_bridge.fcstd_generator import FreeCADGenerator
from constraints.architectural_constraints import ArchitecturalConstraints
class FloorPlanGeneratorPlaceholder:
    def __init__(self):
        print("FloorPlanGeneratorPlaceholder initialized")
        # self.lora_trainer = LoRATrainer() # Would load trained model here
        # self.constraint_checker = ArchitecturalConstraints()

    def create_site_mask(self, width_grids, height_grids):
        print(f"Placeholder: Creating site mask for {width_grids}x{height_grids} grids")
        mask = np.ones((256, 256), dtype=np.uint8) * 255 # Dummy white image
        # Draw a rectangle representing the site
        cv2.rectangle(mask, (10,10), (10 + width_grids*10, 10 + height_grids*10), 0, -1) # Black site area
        return mask

    def generate_plan(self, site_mask_image, prompt):
        print(f"Placeholder: Generating plan with prompt: {prompt}")
        # Dummy raw plan (e.g., a simple image with some features)
        # In reality, this calls SD inference with LoRA model
        # The output should be a multi-channel image as per training data (Walls, Openings, Stairs, Rooms)
        raw_plan_rgba = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
        # Simulate some structure
        cv2.rectangle(raw_plan_rgba, (30,30), (200,200), (0,0,0,255), 2) # Walls in Red channel (simulated)
        raw_plan_rgba[50:60, :, 0] = 255
        raw_plan_rgba[:, 80:90, 0] = 255
        return raw_plan_rgba # Return as RGBA HWC format

    def validate_constraints(self, raw_plan_image):
        print("Placeholder: Validating constraints")
        # Dummy validated plan (could be same as raw or slightly modified)
        # This would call the CP-SAT solver.
        # Output from CP-SAT solver is a dictionary of grids.
        # Need to convert this back to an image for display / further processing.
        # For now, let's assume it returns an image compatible with display.
        if raw_plan_image is None:
            return np.zeros((256,256,3), dtype=np.uint8) # Return blank image if error
        
        # Simulate that the validated plan is similar to raw_plan but maybe as BGR for st.image
        validated_plan_display = raw_plan_image[:,:,:3] # Take RGB from RGBA for display
        # Ensure the array is contiguous and in the correct format for OpenCV
        validated_plan_display = np.ascontiguousarray(validated_plan_display, dtype=np.uint8)
        # Add some green lines to show validation happened
        cv2.line(validated_plan_display, (0,0), (255,255), (0,255,0), 1)
        return validated_plan_display

    def to_svg(self, validated_plan_image_or_data):
        print("Placeholder: Converting to SVG")
        # Dummy SVG data
        return "<svg width=\"100\" height=\"100\"><circle cx=\"50\" cy=\"50\" r=\"40\" stroke=\"green\" stroke-width=\"4\" fill=\"yellow\" /></svg>"

    def to_png_bytes(self, plan_image_np_array):
        print("Placeholder: Converting to PNG bytes")
        is_success, buffer = cv2.imencode(".png", plan_image_np_array)
        if is_success:
            return buffer.tobytes()
        return None
    
    def full_pipeline(self, width, height, rooms, style):
        """ Placeholder for the full generation pipeline for performance testing """
        print("Running full_pipeline placeholder...")
        site_mask = self.create_site_mask(width, height)
        prompt = f"site_size_{width}x{height}, rooms_{rooms}, style_{style}, japanese_house"
        raw_plan = self.generate_plan(site_mask, prompt)
        validated_plan = self.validate_constraints(raw_plan)
        # svg_data = self.to_svg(validated_plan)
        # freecad_result = FreeCADGeneratorPlaceholder().create_3d_model_placeholder(validated_plan, {'site_grid_size': (width, height)}, f"outputs/freecad/model_{width}x{height}.FCStd")
        print("Full_pipeline placeholder finished.")
        return {
            'site_mask': site_mask,
            'raw_plan': raw_plan,
            'validated_plan': validated_plan,
            # 'svg_data': svg_data,
            # 'freecad_result': freecad_result
        }

class FreeCADGeneratorPlaceholder:
    def __init__(self):
        print("FreeCADGeneratorPlaceholder initialized")

    def create_3d_model(self, validated_plan_data, metadata, output_path):
        print(f"Placeholder: Creating 3D model at {output_path}")
        # Dummy FreeCAD result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create a dummy file to simulate FCStd
        with open(output_path, 'w') as f:
            f.write("This is a dummy FreeCAD file.")
        return {
            'fcstd_path': output_path,
            'components': {'walls': 5, 'openings': 2, 'stairs': 1, 'floors': 1}
        }
    
class FloorPlanApp:
    def __init__(self):
        self.generator = FloorPlanGenerator()
        self.freecad_gen = FreeCADGenerator()
        self.constraints = ArchitecturalConstraints()
        
        if 'generated' not in st.session_state:
            st.session_state.generated = False
            st.session_state.svg_data = ""
            st.session_state.freecad_path = ""
            st.session_state.plan_image = None # This should be the image to display
            st.session_state.raw_plan_image = None

    def run(self):
        st.set_page_config(
            page_title="910mmグリッド住宅プラン生成",
            page_icon="🏠",
            layout="wide"
        )
        
        st.title("🏠 AI住宅プラン生成システム")
        st.write("910mm/455mmグリッドベースの住宅平面図を自動生成し、FreeCADで編集可能な3Dモデルを作成")
        
        with st.sidebar:
            st.header("📏 敷地設定")
            width_grids = st.number_input("横幅（グリッド数）", min_value=6, max_value=20, value=11, help="1グリッド = 910mm")
            height_grids = st.number_input("奥行き（グリッド数）", min_value=6, max_value=20, value=10, help="1グリッド = 910mm")
            st.write(f"実寸法: {width_grids * 0.91:.1f}m × {height_grids * 0.91:.1f}m")
            st.write(f"敷地面積: {width_grids * height_grids * 0.91 * 0.91:.1f}㎡")
            
            with st.expander("詳細設定"):
                room_count = st.selectbox("部屋数", ["3LDK", "4LDK", "5LDK"], index=1)
                style = st.selectbox("スタイル", ["standard", "modern", "traditional"])
                
            generate_btn = st.button("🎯 平面図生成", type="primary")
        
        # Main area for results
        results_col = st.container() # Use a container to manage result display area

        if generate_btn:
            with st.spinner("平面図を生成中..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                self.generate_floorplan(width_grids, height_grids, room_count, style, progress_bar, status_text)
        
        if st.session_state.generated:
            with results_col:
                self.show_results(st.session_state.plan_image, st.session_state.svg_data)
                self.show_download_options()
        else:
            with results_col:
                st.info("上記で設定を入力し、「平面図生成」ボタンを押してください。")

    def generate_floorplan(self, width, height, rooms_str, style, progress_bar, status_text):
        """Generate floor plan using actual AI implementations"""
        try:
            status_text.text("敷地マスクを生成中...")
            progress_bar.progress(10)
            site_mask = self.generator.create_site_mask(width, height)
            st.session_state.site_mask_image_for_display = site_mask # For display if needed

            # 2. Generate plan using trained LoRA model
            status_text.text("AI平面図を生成中...")
            progress_bar.progress(30)
            prompt = f"site_size_{width}x{height}, rooms_{rooms_str}, style_{style}, japanese_house"
            raw_plan = self.generator.generate_plan(site_mask, prompt)
            st.session_state.raw_plan_image = raw_plan # Store for potential display
            
            status_text.text("建築制約をチェック中...")
            progress_bar.progress(60)
            
            try:
                validated_plan_data = self.constraints.validate_and_fix(raw_plan)
                
                if validated_plan_data is None:
                    status_text.warning("制約チェックに失敗しました。生の生成結果を使用します。")
                    validated_plan_display = self.generator.validate_constraints(raw_plan)
                else:
                    # We need to convert it to an image for display
                    validated_plan_display = self.convert_validated_data_to_image(validated_plan_data)
            except Exception as e:
                status_text.warning(f"制約システムエラー: {str(e)}。簡易チェックを使用します。")
                validated_plan_display = self.generator.validate_constraints(raw_plan)
            
            st.session_state.plan_image = validated_plan_display
            
            status_text.text("ベクタ図面を作成中...")
            progress_bar.progress(80)
            svg_data = self.generator.to_svg(validated_plan_display)
            st.session_state.svg_data = svg_data
            
            status_text.text("3Dモデルを生成中...")
            
            if 'validated_plan_data' in locals() and validated_plan_data is not None:
                model_data = validated_plan_data
            else:
                # Create simplified data from raw plan
                # Assuming raw_plan is a 3-channel image from the current AI model
                # We'll primarily use the first channel (e.g., for walls)
                # and provide empty or default for others as they are not directly generated as separate channels by SD
                model_data = {
                    'walls': np.zeros(raw_plan.shape[:2], dtype=int),
                    'openings': np.zeros(raw_plan.shape[:2], dtype=int), # Placeholder
                    'stairs': np.zeros(raw_plan.shape[:2], dtype=int),   # Placeholder
                    'rooms': np.zeros(raw_plan.shape[:2], dtype=int),    # Placeholder
                }
                if raw_plan.ndim == 3 and raw_plan.shape[2] > 0:
                    model_data['walls'] = (raw_plan[:,:,0] > 128).astype(int)
                if raw_plan.ndim == 3 and raw_plan.shape[2] > 1:
                    # If you expect openings on the second channel, uncomment and adjust
                    # model_data['openings'] = (raw_plan[:,:,1] > 128).astype(int)
                    pass # Placeholder for now
                if raw_plan.ndim == 3 and raw_plan.shape[2] > 2:
                    # If you expect stairs on the third channel, uncomment and adjust
                    # model_data['stairs'] = (raw_plan[:,:,2] > 128).astype(int)
                    pass # Placeholder for now
                # Rooms (alpha channel) are not expected from a 3-channel image
            
            metadata = {
                'site_grid_size': (width, height),
                'room_count': rooms_str,
                'style': style,
                'total_area_sqm': width * height * 0.91 * 0.91
            }
            
            freecad_result = self.freecad_gen.create_3d_model(
                model_data, 
                metadata,
                f"outputs/freecad/model_{width}x{height}.FCStd"
            )
            
            if freecad_result and 'fcstd_path' in freecad_result:
                st.session_state.freecad_path = freecad_result['fcstd_path']
            else:
                st.session_state.freecad_path = f"outputs/freecad/model_{width}x{height}.FCStd"
                
            st.session_state.metadata = metadata
            
            progress_bar.progress(100)
            st.session_state.generated = True
            status_text.success("✅ 生成完了！")
            
            # Force rerun to display results outside the generate_btn block
            st.rerun()

        except Exception as e:
            st.error(f"生成エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            status_text.error("❌ 生成失敗")
            st.session_state.generated = False # Ensure results are not shown on error

    def convert_validated_data_to_image(self, validated_data):
        """Convert validated data dictionary to displayable image"""
        if validated_data is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
            
        # Create a blank RGB image
        height, width = validated_data['walls'].shape
        display_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if 'walls' in validated_data:
            display_image[:, :, 2] = validated_data['walls'] * 255  # Walls in blue
            
        if 'rooms' in validated_data:
            room_display = np.zeros_like(validated_data['rooms'])
            for room_id in range(1, 10):  # Room IDs 1-9
                room_display[validated_data['rooms'] == room_id] = 100 + (room_id * 15)
            display_image[:, :, 1] = room_display  # Rooms in green
            
        if 'stairs_1f' in validated_data:
            display_image[:, :, 0] = validated_data['stairs_1f'] * 255  # Stairs in red
            
        # Ensure the array is contiguous and in the correct format
        display_image = np.ascontiguousarray(display_image, dtype=np.uint8)
        
        # デバッグ用の緑の斜線を削除
        # cv2.line(display_image, (0, 0), (width-1, height-1), (0, 255, 0), 1)
        
        return display_image
        
    def analyze_plan(self, plan_image):
        """Analyze plan details from metadata and image"""
        metadata = getattr(st.session_state, 'metadata', {})
        
        room_count = metadata.get('room_count', "N/A")
        total_area = metadata.get('total_area_sqm', "N/A")
        
        # In a real implementation, this would analyze the plan image
        
        return {
            "estimated_rooms": room_count,
            "total_area_sqm": total_area,
            "warnings": []
        }

    def show_results(self, plan_image_to_display, svg_data):
        st.success("平面図が正常に生成されました！")
        tab1, tab2, tab3 = st.tabs(["🖼️ プレビュー", "📐 詳細情報", "🔧 編集オプション"])
        
        with tab1:
            if plan_image_to_display is not None:
                # Assuming plan_image_to_display is a NumPy array (BGR format for OpenCV)
                st.image(plan_image_to_display, caption="生成された平面図", use_column_width=True)
            else:
                st.warning("プレビュー画像が利用できません。")
            # For debugging, show raw plan if available
            # if st.session_state.raw_plan_image is not None:
            #     st.image(st.session_state.raw_plan_image, caption="Raw AI Output (RGBA)", use_column_width=True)
            if st.session_state.site_mask_image_for_display is not None:
                 st.image(st.session_state.site_mask_image_for_display, caption="敷地マスク", use_column_width=True)

        with tab2:
            if hasattr(st.session_state, 'metadata') and st.session_state.metadata:
                room_info = self.analyze_plan(st.session_state.metadata) # Pass metadata here
                st.json(room_info)
            else:
                st.write("詳細情報はありません（メタデータが見つかりません）。")
            
        with tab3:
            st.write("FreeCADで編集可能なファイルが生成されました（outputs/freecad/ フォルダ内）。")
            st.write("- 壁の厚み・高さはスプレッドシートからパラメトリックに変更可能（予定）。")
            st.write("- 現状の3Dモデルはプレースホルダーが多く含まれます。")
            st.write("- DXF, STEP等でのエクスポートも可能です（editing_features.py経由）。")

    def show_download_options(self):
        st.header("📥 ダウンロード")
        # PNG画像
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.plan_image is not None:
                png_bytes = self.generator.to_png_bytes(st.session_state.plan_image) # Use display image
                if png_bytes:
                    st.download_button(
                        label="🖼️ PNG 画像",
                        data=png_bytes,
                        file_name="floorplan_validated.png",
                        mime="image/png"
                    )
                else:
                    st.error("PNG変換エラー")
            else:
                st.button("🖼️ PNG 画像", disabled=True)
        
        with col2:
            if st.session_state.svg_data:
                st.download_button(
                    label="📄 SVG 図面",
                    data=st.session_state.svg_data,
                    file_name="floorplan.svg",
                    mime="image/svg+xml"
                )
            else:
                st.button("📄 SVG 図面", disabled=True)
        
        with col3:
            if st.session_state.freecad_path and os.path.exists(st.session_state.freecad_path):
                with open(st.session_state.freecad_path, 'rb') as fp:
                    fcstd_bytes = fp.read()
                st.download_button(
                    label="🎯 FreeCAD (.FCStd)",
                    data=fcstd_bytes,
                    file_name=os.path.basename(st.session_state.freecad_path),
                    mime="application/octet-stream" # Generic binary type
                )
            else:
                st.button("🎯 FreeCAD (.FCStd)", disabled=True)

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs("outputs/generated", exist_ok=True)
    os.makedirs("outputs/svg", exist_ok=True)
    os.makedirs("outputs/dxf", exist_ok=True)
    os.makedirs("outputs/freecad", exist_ok=True)
    
    app = FloorPlanApp()
    app.run()                                