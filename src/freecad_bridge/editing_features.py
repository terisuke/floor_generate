import FreeCAD as App
import Part
import importDXF # For DXF export, part of FreeCAD standard modules
import Import # For STEP/IGES export, part of FreeCAD standard modules
import os

class EditingFeatures:
    def __init__(self, fcstd_doc_path=None):
        if fcstd_doc_path and os.path.exists(fcstd_doc_path):
            self.doc = App.openDocument(fcstd_doc_path)
            App.setActiveDocument(self.doc.Name)
        elif App.ActiveDocument:
            self.doc = App.ActiveDocument
        else:
            print("Error: No active FreeCAD document and no document path provided.")
            self.doc = None # Or raise an error

    def setup_parametric_features(self):
        """パラメトリック編集機能をセットアップ (Requirement 7.2)
           - 壁の高さと厚みのグローバルパラメータを作成
           - 壁オブジェクトをこれらのパラメータにリンク
        """
        if not self.doc:
            print("No document loaded to set up parametric features.")
            return

        print("Setting up parametric features...")
        
        # 1. 寸法パラメータ作成 (例: Spreadsheetオブジェクトを使用)
        # Check if spreadsheet already exists
        spreadsheet_name = "ModelParameters"
        param_sheet = self.doc.getObject(spreadsheet_name)
        if not param_sheet:
            param_sheet = self.doc.addObject("Spreadsheet::Sheet", spreadsheet_name)
        
        # 壁の高さ (WallHeight)
        param_sheet.set('A1', 'WallHeight_label')
        param_sheet.set('B1', '2400') # Default value in mm
        param_sheet.setAlias('B1', 'WallHeight')
        param_sheet.setDisplayUnit('B1', 'mm')

        # 壁の厚み (WallThickness)
        param_sheet.set('A2', 'WallThickness_label')
        param_sheet.set('B2', '105') # Default value in mm
        param_sheet.setAlias('B2', 'WallThickness')
        param_sheet.setDisplayUnit('B2', 'mm')

        # 2. 壁とパラメータをリンク
        for obj in self.doc.Objects:
            if obj.isDerivedFrom("Arch::Wall") or "Wall" in obj.Label: # Check if it's an Arch Wall
                try:
                    # 壁の高さ (Height property for ArchWall)
                    obj.setExpression('Height', f'{spreadsheet_name}.WallHeight')
                    # 壁の厚み (typically Width or Thickness, ArchWall uses 'Width')
                    # Check if 'Width' property exists for standard walls. Might need specific handling if using sketches for walls.
                    if hasattr(obj, 'Width'):
                         obj.setExpression('Width', f'{spreadsheet_name}.WallThickness')
                    elif hasattr(obj, 'Thickness'): # Some custom walls might use Thickness
                         obj.setExpression('Thickness', f'{spreadsheet_name}.WallThickness')
                    print(f"Linked parameters for wall: {obj.Label}")
                except Exception as e:
                    print(f"Error linking parameters for {obj.Label}: {e}")
        
        self.doc.recompute()
        print("Parametric features set up.")

    def extract_wall_centerlines(self):
        """壁オブジェクトから中心線を抽出 (Placeholder)
           This is a complex task depending on how walls were created.
           If walls are based on Draft Wires, we can get their points.
           Returns a list of lines, where each line is [App.Vector_start, App.Vector_end]
        """
        print("Extracting wall centerlines (placeholder)...")
        wall_centerlines = []
        for obj in self.doc.Objects:
            if obj.isDerivedFrom("Arch::Wall") and hasattr(obj, 'Base') and obj.Base and obj.Base.isDerivedFrom("Draft::Wire"):
                points = obj.Base.Shape.Vertexes
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        wall_centerlines.append([points[i].Point, points[i+1].Point])
                    # If the wire was closed and represented a loop, the last segment is missing.
                    # This needs to be handled based on how walls are defined.
        print(f"Extracted {len(wall_centerlines)} wall centerline segments (placeholder logic).")
        return wall_centerlines

    def add_dimensional_constraints(self, sketch_object):
        """スケッチに寸法制約を追加 (Placeholder)"""
        print("Adding dimensional constraints to sketch (placeholder)...")
        # This is highly dependent on the sketch content and desired constraints.
        # Example: Add distance constraints between parallel lines or length of lines.
        # For MVP, this is complex and likely requires manual user input in FreeCAD or a more defined plan.
        pass

    def make_sketches_editable(self):
        """スケッチを編集可能にする (Requirement 7.2)
           - 平面図の基準スケッチを作成 (if not exists or if we want a new master sketch)
           - 壁の中心線をスケッチに追加
           - 寸法制約追加 (placeholder)
        """
        if not self.doc:
            print("No document loaded to make sketches editable.")
            return

        print("Making sketches editable...")
        # Create or find a base sketch for the floor plan
        sketch_name = "FloorPlanSketch"
        base_sketch = self.doc.getObject(sketch_name)
        if not base_sketch:
            base_sketch = self.doc.addObject("Sketcher::SketchObject", sketch_name)
        else:
            # Clear existing geometry if we are regenerating it
            # base_sketch.clearGeometry()
            pass # For now, append or assume it's managed

        # 壁の中心線をスケッチに追加
        wall_centerlines = self.extract_wall_centerlines()
        # The sketch expects indices of points, not Part.LineSegment directly for addGeometry in some contexts.
        # Or it can take Part.LineSegment objects.
        for line_segment_points in wall_centerlines:
            try:
                # Add line segments to sketch. Ensure points are 2D for XY sketch plane.
                # Assuming sketch is on XY plane.
                p1 = App.Vector(line_segment_points[0].x, line_segment_points[0].y, 0)
                p2 = App.Vector(line_segment_points[1].x, line_segment_points[1].y, 0)
                line = Part.LineSegment(p1, p2)
                base_sketch.addGeometry(line, False) # False means not for construction
            except Exception as e:
                print(f"Error adding line to sketch: {e}")

        # 寸法制約追加 (Placeholder for actual constraint logic)
        self.add_dimensional_constraints(base_sketch)
        
        self.doc.recompute()
        print(f"Sketch '{sketch_name}' prepared for editing.")


    def export_for_editing(self, export_formats=['step', 'iges', 'dxf']):
        """他のCADソフト用フォーマットでエクスポート (Requirement 7.2)"""
        if not self.doc:
            print("No document loaded to export.")
            return {}
            
        print(f"Exporting document {self.doc.Name}...")
        exports = {}
        doc_path = self.doc.FileName
        if not doc_path:
            # If document not saved, use a temporary name in a writable directory
            # This needs care for where to save temp files.
            # For now, assume doc has been saved or export might fail to name correctly.
            base_name = "exported_floorplan"
            # Try to use a known writable path, e.g., where the script is or /tmp
            # This part is environment-dependent.
            # Let's assume the output directory from main script is available for simplicity.
            temp_dir = "outputs/temp_exports/" 
            os.makedirs(temp_dir, exist_ok=True)
            doc_path = os.path.join(temp_dir, base_name + ".FCStd") # Dummy path for naming
            print(f"Warning: Document not saved. Using temporary base name for export: {base_name}")

        base_export_path = os.path.splitext(doc_path)[0]

        # Ensure objects to export are selected or specify all
        objects_to_export = self.doc.Objects # Export all objects in the document
        if not objects_to_export:
            print("No objects found in the document to export.")
            return {}

        for fmt in export_formats:
            output_path = f"{base_export_path}.{fmt.lower()}"
            try:
                if fmt.lower() == 'step':
                    Import.export(objects_to_export, output_path)
                    exports['step'] = output_path
                    print(f"Exported to STEP: {output_path}")
                elif fmt.lower() == 'iges':
                    Import.export(objects_to_export, output_path)
                    exports['iges'] = output_path
                    print(f"Exported to IGES: {output_path}")
                elif fmt.lower() == 'dxf':
                    # DXF export might need specific view or 2D projection of 3D objects
                    # For Arch objects, usually, a 2D projection (like a plan view) is exported.
                    # Simple DXF export of 3D objects might not be ideal.
                    # Placeholder: export all objects as is. May need a Shape2DView for proper DXF plan.
                    importDXF.export(objects_to_export, output_path)
                    exports['dxf'] = output_path
                    print(f"Exported to DXF: {output_path}")
                else:
                    print(f"Unsupported export format: {fmt}")
            except Exception as e:
                print(f"Error exporting to {fmt}: {e}")
        
        return exports

# Example Usage:
# if __name__ == '__main__':
#     # This script needs to be run within FreeCAD's Python console or with FreeCAD libs properly set up.
#     # Assuming a document is open or a path is provided.
#     # fc_doc_path = "path_to_your_model.FCStd" 
#     # editor_features = EditingFeatures(fc_doc_path)
#     
#     # If a document is already active in FreeCAD GUI:
#     if App.ActiveDocument:
#         editor_features = EditingFeatures() # Uses active document
#         editor_features.setup_parametric_features()
#         editor_features.make_sketches_editable()
#         exported_files = editor_features.export_for_editing(['step', 'dxf'])
#         print("Exported files:", exported_files)
#     else:
#         print("No active FreeCAD document. Open a model first or provide a path.") 