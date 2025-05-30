import os
import sys
import numpy as np
import cv2  # Add missing import

# FreeCAD imports with error handling
try:
    import FreeCAD as App
    import Draft, Arch, Part
except ImportError:
    print("Warning: FreeCAD not available. Using placeholder mode.")
    App = None

class FreeCADGenerator:
    def __init__(self):
        self.doc = None
        self.wall_height = 2400  # mm
        self.wall_thickness = 105  # mm (在来工法標準)
        self.grid_size_mm = 910 # 主グリッドサイズ

    def create_scale_converter(self, metadata):
        """グリッド数を実寸法(mm)に変換するためのスケール情報を設定。
           metadataから敷地のグリッドサイズを取得し、1グリッドあたりのmmを返す。
           現状は主グリッド910mm固定とする。
        """
        # site_grid_size_pixels = metadata.get('site_grid_size_pixels', (256, 256)) # Target image size
        # site_grid_count = metadata.get('site_grid_size', (11, 10)) # Example grid counts
        # Assuming primary grid is 910mm as per requirements
        # The conversion factor is simply the grid size in mm.
        return {
            'primary': self.grid_size_mm, # 910mm
            'secondary': self.grid_size_mm / 2 # 455mm
        }

    def extract_wall_contours(self, validated_plan_grids):
        """検証済みプランの壁グリッドから壁の輪郭を抽出 (Placeholder)
           validated_plan_grids: 壁の位置を示す2D numpy配列 (1が壁、0が空間)
           戻り値: 輪郭点のリストのリスト [[(x1,y1), (x2,y2), ...], ...]
                  座標はグリッド座標。
        """
        print("Extracting wall contours (placeholder)...")
        # This is a complex computer vision task. For MVP, use a simplified approach.
        # Find contours using OpenCV on the wall grid.
        # Ensure validated_plan_grids is uint8 for findContours
        if validated_plan_grids.dtype != np.uint8:
            wall_grid_uint8 = (validated_plan_grids * 255).astype(np.uint8)
        else:
            wall_grid_uint8 = validated_plan_grids
        
        contours, hierarchy = cv2.findContours(wall_grid_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wall_contours_pixel_coords = []
        for contour in contours:
            # Approximate contour to reduce points, but might lose precision for grid lines
            # epsilon = 0.001 * cv2.arcLength(contour, True) # Very small epsilon to keep lines straight
            # approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            # For grid-based, we might not need approximation if contours are already grid-aligned
            # Convert contour points from [[x,y]] to [(x,y)]
            wall_contours_pixel_coords.append([tuple(point[0]) for point in contour])
        
        print(f"Found {len(wall_contours_pixel_coords)} wall contours.")
        # These are pixel coordinates; they should map directly to grid coordinates if the input grid is 1-to-1 with pixels.
        # If the input `validated_plan_grids` is already a grid (not an image), then these are grid coordinates.
        return wall_contours_pixel_coords

    def detect_openings(self, validated_plan_grids):
        """検証済みプランから開口部位置を検出 (Placeholder)
           validated_plan_grids: ここでは、AI出力のチャンネル分離された画像(の開口部チャンネル)を想定。
                                 もしくは、壁グリッドと部屋情報から推測。
           戻り値: 開口部の位置とタイプのリスト [{type:'door', pos:(gx,gy,gw,gh)}, ...]
        """
        print("Detecting openings (placeholder)...")
        # This would ideally come from the AI's output channels or sophisticated analysis.
        # Placeholder: Return a few dummy openings.
        # Assuming `validated_plan_grids` is the multichannel AI output where channel 1 is openings.
        openings = []
        if validated_plan_grids.ndim == 3 and validated_plan_grids.shape[2] > 1:
            opening_channel = (validated_plan_grids[:,:,1] * 255).astype(np.uint8) # Assuming channel 1 is openings, normalized
            contours, _ = cv2.findContours(opening_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 2: # Min area for an opening (grid cells)
                    x, y, w, h = cv2.boundingRect(cnt)
                    # For simplicity, classify all as doors for now
                    openings.append({'type': 'door', 'pos_grid': (x, y, w, h), 'grid_rect': (x,y,w,h)})
        print(f"Detected {len(openings)} dummy openings.")
        return openings

    def detect_stairs(self, validated_plan_grids):
        """検証済みプランから階段位置を検出 (Placeholder)
           validated_plan_grids: AI出力のチャンネル分離された画像(の階段チャンネル)を想定。
           戻り値: 階段位置と寸法のリスト [{pos_grid:(gx,gy), length_grid:L, width_grid:W}, ...]
        """
        print("Detecting stairs (placeholder)...")
        # Placeholder: return dummy stair info
        stairs_info = []
        if validated_plan_grids.ndim == 3 and validated_plan_grids.shape[2] > 2:
            stair_channel = (validated_plan_grids[:,:,2] * 255).astype(np.uint8) # Assuming channel 2 is stairs
            contours, _ = cv2.findContours(stair_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 3: # Min area for stairs (grid cells)
                    x,y,w,h = cv2.boundingRect(cnt)
                    # Determine length and width based on orientation (simplified)
                    length_grid = max(w,h)
                    width_grid = min(w,h)
                    stairs_info.append({'x_grid': x, 'y_grid': y, 'length_grid': length_grid, 'width_grid': width_grid})
        print(f"Detected {len(stairs_info)} dummy stairs.")
        return stairs_info

    def create_walls(self, validated_plan_grids, grid_to_mm_scale):
        """壁を生成"""
        print("Creating walls in FreeCAD...")
        walls_fc_objects = []
        wall_contours_grid = self.extract_wall_contours(validated_plan_grids['walls']) # Assuming grids_dict has 'walls' key

        for contour_grid in wall_contours_grid:
            if len(contour_grid) < 2: continue # Need at least two points for a wire

            real_points = []
            for point_grid in contour_grid:
                # Assuming grid coordinates are top-left origin, Y positive downwards
                # FreeCAD Y is typically upwards, so invert Y if necessary based on source
                # Let's assume grid coords are (col, row)
                real_x = point_grid[0] * grid_to_mm_scale['primary']
                real_y = point_grid[1] * grid_to_mm_scale['primary'] 
                real_points.append(App.Vector(real_x, real_y, 0))
            
            if not real_points:
                continue

            # If contour is not closed, add first point to end to close it for wall creation
            if real_points[0].distanceToPoint(real_points[-1]) > 1e-3: # If not already closed
                 real_points.append(real_points[0])

            try:
                wire = Draft.makeWire(real_points, closed=True) # Ensure wire is closed
                self.doc.recompute() 
                if wire.Shape.Wires:
                    wall = Arch.makeWall(
                        wire, 
                        length=None, # Length is derived from wire
                        width=self.wall_thickness,
                        height=self.wall_height,
                        align="Center" # Align wall to center of the wire
                    )
                    wall.Label = f"Wall_{len(walls_fc_objects)+1}"
                    # wall.Material = "Concrete" # Example material
                    self.doc.addObject(wall)
                    walls_fc_objects.append(wall)
                else:
                    print(f"Warning: Could not create valid wire from points: {real_points}")
            except Exception as e:
                print(f"Error creating wall from wire: {e}. Points: {real_points}")
        
        self.doc.recompute() # Recompute after all walls are added
        print(f"Created {len(walls_fc_objects)} wall objects.")
        return walls_fc_objects

    def find_nearest_wall(self, opening_pos_mm, walls_fc_objects):
        """開口部の位置に最も近い壁を見つける (Placeholder)"""
        # opening_pos_mm: (x, y, w, h) in mm
        if not walls_fc_objects: return None
        # Simplified: return the first wall as a placeholder
        # A real implementation would check distances or ray intersections.
        print("Finding nearest wall (placeholder)...")
        return walls_fc_objects[0]

    def create_door(self, opening_data, wall_fc_object, grid_to_mm_scale):
        """ドア生成"""
        print(f"Creating door for opening: {opening_data}")
        door_width_default = 780  # mm
        door_height_default = 2000  # mm

        # Use opening_data['pos_grid'] and opening_data['grid_rect']
        grid_rect = opening_data['grid_rect'] # (gx, gy, gw, gh)
        pos_x_mm = grid_rect[0] * grid_to_mm_scale['primary'] + (grid_rect[2] * grid_to_mm_scale['primary']) / 2
        pos_y_mm = grid_rect[1] * grid_to_mm_scale['primary'] + (grid_rect[3] * grid_to_mm_scale['primary']) / 2

        opening_width_mm = grid_rect[2] * grid_to_mm_scale['primary']
        opening_height_mm = door_height_default # Assume standard height

        # Create a sketch for the window geometry on the wall's plane
        # This is complex. For MVP, let's use Arch.makeWindow with a simple box.
        # The position and orientation need to be relative to the wall.
        
        # Create a simple box for the opening shape
        # The box needs to be placed correctly on the wall face.
        # This requires finding the wall's plane and orientation.
        # Simplified approach: Use Arch.addWindow, which tries to handle placement.

        try:
            # Create a sketch for the door
            sketch = self.doc.addObject('Sketcher::SketchObject', 'DoorSketch')
            # Define rectangle for door opening based on opening_width_mm, opening_height_mm
            # The sketch needs to be on the wall's face and positioned correctly.
            # This is non-trivial. Placeholder: create a window, which is simpler.
            # A simple way to make a window is to give its dimensions and a host wall.
            # The `Arch.makeWindow` function is usually for predefined window types.
            # Using `Arch.addWindow` (or `Arch.removeComponents`) with a subtraction solid is more general.

            # Let's try to create a subtraction box.
            # Base of the door opening (assuming bottom of wall is z=0)
            # Position needs to be on the wall line.
            # This is simplified: Assumes wall is along X or Y axis and opening pos is on the wall center line.
            # We need more info about wall orientation to place this correctly.
            
            # Placeholder: Create a box and try to subtract it.
            # This is very basic and likely won't align correctly without wall geometry details.
            box = self.doc.addObject("Part::Box", "DoorOpeningShape")
            box.Length = opening_width_mm
            box.Width = self.wall_thickness + 20 # Make it slightly thicker than wall to ensure cut
            box.Height = opening_height_mm
            # Position the box - this is the hardest part without wall orientation
            # Assuming opening_pos_mm is the center of the opening for now.
            box.Placement = App.Placement(App.Vector(pos_x_mm - opening_width_mm/2, 
                                                   pos_y_mm - (self.wall_thickness + 20)/2, 
                                                   0), App.Rotation(App.Vector(0,0,1),0))
            
            self.doc.recompute()
            # Arch.removeComponents(box, wall_fc_object) # This should work if box is correctly placed
            # A more robust way for windows/doors is using ArchWindow object
            win = Arch.makeWindowPreset("Simple", width=opening_width_mm, height=opening_height_mm)
            self.doc.addObject(win)
            # The host wall and placement on the wall is tricky. 
            # Arch.addWindow(win, wall_fc_object) might work if `win` is prepared correctly.
            # For MVP, this is a known difficult part.
            print(f"Placeholder door created. Manual placement in FreeCAD might be needed.")
            return win # Or box, or whatever object represents the opening

        except Exception as e:
            print(f"Error creating door: {e}")
            return None

    def create_openings(self, validated_plan_grids_dict, walls_fc_objects, grid_to_mm_scale):
        """開口部（ドア・窓）を生成"""
        print("Creating openings in FreeCAD...")
        openings_fc_objects = []
        opening_data_list = self.detect_openings(validated_plan_grids_dict['openings']) # Assuming 'openings' channel

        for opening_data in opening_data_list:
            # Opening data format: {'type': 'door', 'grid_rect': (gx, gy, gw, gh)}
            
            # Simplified: find a wall to host this. This logic needs improvement.
            if not walls_fc_objects:
                print("No walls to host openings.")
                continue
            
            # Convert grid_rect to mm coordinates to find nearest wall (very simplified)
            gx, gy, gw, gh = opening_data['grid_rect']
            opening_center_x_mm = (gx + gw/2) * grid_to_mm_scale['primary']
            opening_center_y_mm = (gy + gh/2) * grid_to_mm_scale['primary']
            opening_pos_mm_approx = (opening_center_x_mm, opening_center_y_mm, 
                                     gw * grid_to_mm_scale['primary'], gh*grid_to_mm_scale['primary'])

            nearest_wall = self.find_nearest_wall(opening_pos_mm_approx, walls_fc_objects)

            if nearest_wall:
                if opening_data['type'] == 'door':
                    fc_opening = self.create_door(opening_data, nearest_wall, grid_to_mm_scale)
                else: # Assume window
                    # fc_opening = self.create_window(opening_data, nearest_wall, grid_to_mm_scale) # Placeholder
                    print(f"Window creation placeholder for: {opening_data}")
                    fc_opening = None 
                
                if fc_opening:
                    openings_fc_objects.append(fc_opening)
            else:
                print(f"Could not find a host wall for opening at grid {opening_data['grid_rect']}")
        
        self.doc.recompute()
        print(f"Created {len(openings_fc_objects)} opening objects (placeholders).")
        return openings_fc_objects

    def create_stairs(self, validated_plan_grids_dict, grid_to_mm_scale):
        """階段生成"""
        print("Creating stairs in FreeCAD...")
        stairs_fc_objects = []
        stair_data_list = self.detect_stairs(validated_plan_grids_dict['stairs']) # Assuming 'stairs' channel

        for stair_data in stair_data_list:
            # stair_data: {'x_grid': gx, 'y_grid': gy, 'length_grid': L, 'width_grid': W}
            stair_width_mm = stair_data['width_grid'] * grid_to_mm_scale['primary']
            stair_length_mm = stair_data['length_grid'] * grid_to_mm_scale['primary']
            step_height_default = 200  # mm
            num_steps = int(self.wall_height / step_height_default)

            real_x_mm = stair_data['x_grid'] * grid_to_mm_scale['primary']
            real_y_mm = stair_data['y_grid'] * grid_to_mm_scale['primary']

            try:
                stairs = Arch.makeStairs(
                    length=stair_length_mm,
                    width=stair_width_mm,
                    height=self.wall_height,
                    steps=num_steps
                )
                stairs.Placement = App.Placement(App.Vector(real_x_mm, real_y_mm, 0), App.Rotation(App.Vector(0,0,1),0))
                stairs.Label = f"Stairs_{len(stairs_fc_objects)+1}"
                self.doc.addObject(stairs)
                stairs_fc_objects.append(stairs)
            except Exception as e:
                print(f"Error creating stairs: {e}")

        self.doc.recompute()
        print(f"Created {len(stairs_fc_objects)} stair objects.")
        return stairs_fc_objects

    def create_floors(self, validated_plan_grids_dict, grid_to_mm_scale):
        """フロア（スラブ）生成 (Placeholder)"""
        print("Creating floors (placeholder)...")
        # This would typically use the room boundaries or overall building footprint.
        # For MVP, create a simple slab based on the largest detected contour or overall site extent.
        # Assuming 'rooms' channel or overall wall outline can define floor area.
        floors_fc_objects = []
        
        # Placeholder: Create a slab covering the approximate site area
        # This needs to be more sophisticated, e.g., using wall outlines.
        site_footprint_contour_grid = None # Needs to be derived from walls or rooms
        # Example: use the largest wall contour for simplicity
        wall_contours_grid = self.extract_wall_contours(validated_plan_grids_dict['walls'])
        if wall_contours_grid:
            site_footprint_contour_grid = max(wall_contours_grid, key=len) # Simplistic
        
        if site_footprint_contour_grid and len(site_footprint_contour_grid) > 2:
            real_points = []
            for point_grid in site_footprint_contour_grid:
                real_x = point_grid[0] * grid_to_mm_scale['primary']
                real_y = point_grid[1] * grid_to_mm_scale['primary']
                real_points.append(App.Vector(real_x, real_y, 0))
            
            try:
                face = Part.Face(Draft.makeWire(real_points, closed=True))
                slab = self.doc.addObject("Part::Feature","FloorSlab")
                slab.Shape = face.extrude(App.Vector(0,0,-200)) # Extrude downwards for slab
                slab.Label = "FloorSlab_1F"
                self.doc.addObject(slab)
                floors_fc_objects.append(slab)
            except Exception as e:
                print(f"Error creating floor slab: {e}")

        self.doc.recompute()
        print(f"Created {len(floors_fc_objects)} floor objects (placeholders).")
        return floors_fc_objects

    def create_building(self, components_list):
        """建物全体を統合"""
        print("Creating building object...")
        if not App.ActiveDocument:
             print("No active document to create building in.")
             return None
        if App.ActiveDocument != self.doc:
            App.setActiveDocument(self.doc.Name)
            
        building = Arch.makeBuilding(components_list)
        building.Label = "Generated_House"
        self.doc.recompute()
        print("Building object created.")
        return building

    def create_3d_model(self, validated_plan_grids_dict, metadata, output_path):
        """検証済み平面図から3Dモデル生成"""
        print(f"Starting 3D model generation for output: {output_path}")

        if App is None:
            print("Error: FreeCAD module is not available. Cannot create 3D model.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Create a uniquely named dummy file to avoid overwriting if called multiple times
            dummy_fcstd_path = os.path.join(os.path.dirname(output_path), f"dummy_fc_unavailable_{os.path.basename(output_path)}")
            try:
                with open(dummy_fcstd_path, 'w') as f:
                    f.write("FreeCAD not available. This is a placeholder file.")
            except Exception as e_file:
                print(f"Error creating dummy file: {e_file}")
                # If dummy file creation fails, return path that might not exist but follows pattern
                dummy_fcstd_path = output_path # Fallback to original path for return structure
            return {
                'fcstd_path': dummy_fcstd_path,
                'components': {},
                'error': "FreeCAD module not available."
            }

        # 1. 新規文書作成
        # App is not None here, so it's safe to call App.newDocument
        doc_name = f"FloorPlan_{metadata.get('site_grid_size', (0,0))[0]}x{metadata.get('site_grid_size', (0,0))[1]}"
        try:
            self.doc = App.newDocument(doc_name)
        except Exception as e_doc_create: # Catch any potential error during newDocument
            print(f"Error: Exception while creating new FreeCAD document '{doc_name}': {e_doc_create}")
            self.doc = None # Ensure self.doc is None if creation fails

        if self.doc is None:
             print(f"Error: Failed to create new FreeCAD document: '{doc_name}'. App.newDocument might have returned None or raised an exception.")
             os.makedirs(os.path.dirname(output_path), exist_ok=True)
             dummy_fcstd_path = os.path.join(os.path.dirname(output_path), f"dummy_doc_fail_{os.path.basename(output_path)}")
             try:
                 with open(dummy_fcstd_path, 'w') as f:
                     f.write("Failed to create FreeCAD document. This is a placeholder file.")
             except Exception as e_file_doc_fail:
                 print(f"Error creating dummy file on doc creation failure: {e_file_doc_fail}")
                 dummy_fcstd_path = output_path
             return {
                'fcstd_path': dummy_fcstd_path,
                'components': {},
                'error': "Failed to create FreeCAD document."
            }
        
        print(f"Successfully created FreeCAD document: {self.doc.Name}")

        grid_to_mm_scale = self.create_scale_converter(metadata)

        walls_fc = self.create_walls(validated_plan_grids_dict, grid_to_mm_scale)
        openings_fc = self.create_openings(validated_plan_grids_dict, walls_fc, grid_to_mm_scale)
        stairs_fc = self.create_stairs(validated_plan_grids_dict, grid_to_mm_scale)
        floors_fc = self.create_floors(validated_plan_grids_dict, grid_to_mm_scale)

        all_components = []
        if walls_fc: all_components.extend(walls_fc)
        if openings_fc: all_components.extend(openings_fc) # Note: Openings might be part of walls already
        if stairs_fc: all_components.extend(stairs_fc)
        if floors_fc: all_components.extend(floors_fc)
        
        building = self.create_building(all_components)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.doc.saveAs(output_path)
        print(f"3D model saved to {output_path}")

        # App.closeDocument(self.doc.Name) # Close doc if running in batch
        # self.doc = None

        return {
            'fcstd_path': output_path,
            'components': {
                'walls': len(walls_fc) if walls_fc else 0,
                'openings': len(openings_fc) if openings_fc else 0,
                'stairs': len(stairs_fc) if stairs_fc else 0,
                'floors': len(floors_fc) if floors_fc else 0
            }
        }


# Note: This class assumes it's being run within a FreeCAD Python environment
# or that FreeCAD modules can be imported standalone (e.g., via `pip install freecad`).
# Direct execution of this script outside FreeCAD might require specific setup.
# The placeholders for contour/element detection are significant simplifications.
# A full implementation of those would be major sub-projects.  