from ortools.sat.python import cp_model
import numpy as np

class ArchitecturalConstraints:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        # Potentially add solver parameters if needed, e.g., time limits
        # self.solver.parameters.max_time_in_seconds = 10.0

    def image_to_grid(self, floor_plan_image):
        """Converts a floor plan image (presumably RGBA) to a simplified grid.
        This method needs a clear definition of how image channels/values map to grid cell types.
        The requirement (section 6.1) mentions `grid[i, j] == 1` for wall cells.
        Let's assume the 'walls' channel (R) of the floor_plan_image is used.
        Values > threshold (e.g., 128) are walls (1), others are non-walls (0).
        The grid size should match the floor_plan_image dimensions.
        """
        print("Converting image to grid...")
        if floor_plan_image is None or floor_plan_image.ndim < 2:
            raise ValueError("Invalid floor_plan_image provided for grid conversion.")

        # Assuming floor_plan_image is HxWx4 (RGBA) as saved by TrainingDataGenerator
        # and walls are in the first channel (Red)
        # Let's also assume it's normalized to 0-1 range if coming from AI output, then scaled to 0-255.
        # If it's already 0-255, then no need to scale up.
        # For MVP, let's assume the input image is a 2D numpy array where wall pixels are marked (e.g., >0 or specific value)
        # or it's the direct output from the AI (which might be 4-channel image).

        # Placeholder logic: Use the first channel (e.g., walls) and threshold it.
        # This needs to align with the AI output format.
        # If the AI outputs a 4-channel image [0-1], use the first channel.
        if floor_plan_image.ndim == 3 and floor_plan_image.shape[2] >= 1: # HWC or HW(RGBA)
            wall_channel = floor_plan_image[:, :, 0]
        elif floor_plan_image.ndim == 2: # Grayscale image, assume it represents walls
            wall_channel = floor_plan_image
        else:
            raise ValueError(f"Unsupported floor_plan_image format: {floor_plan_image.shape}")

        # Threshold to get binary wall grid. Assuming wall pixels are > 0.5 if image is [0,1]
        # or > 128 if [0,255]. Let's assume [0,1] for now from AI.
        # The requirement uses grid[i,j] == 1 for wall.
        grid = (wall_channel > 0.5).astype(int) 
        print(f"Grid shape: {grid.shape}, Wall cells: {np.sum(grid)}")
        return grid

    def define_variables(self, height, width):
        """Defines CP-SAT variables for walls, rooms, stairs, etc.
           Based on the requirement, variables are needed for:
           - walls[i][j]
           - rooms[i][j] (integer for room ID or type)
           - stairs_1f[i][j], stairs_2f[i][j]
        """
        print("Defining CP-SAT variables...")
        variables = {}
        variables['wall'] = [[self.model.NewBoolVar(f'wall_{i}_{j}') for j in range(width)] for i in range(height)]
        
        # Rooms: 0 for non-room, 1-N for different rooms. Max 10 rooms from requirement.
        # Max room ID will be 9 (0 is non-room).
        variables['rooms'] = [[self.model.NewIntVar(0, 9, f'room_{i}_{j}') for j in range(width)] for i in range(height)]

        # Stairs: 0 for no stair, 1 for stair.
        variables['stairs_1f'] = [[self.model.NewBoolVar(f'stair_1f_{i}_{j}') for j in range(width)] for i in range(height)]
        variables['stairs_2f'] = [[self.model.NewBoolVar(f'stair_2f_{i}_{j}') for j in range(width)] for i in range(height)]
        
        print(f"Variables defined for grid {height}x{width}")
        return variables

    def get_neighbors(self, r, c, height, width, connectivity=4):
        """Returns valid neighbors for a cell (r, c)."""
        neighbors = []
        # 4-connectivity (N, S, E, W)
        if r > 0: neighbors.append((r - 1, c))
        if r < height - 1: neighbors.append((r + 1, c))
        if c > 0: neighbors.append((r, c - 1))
        if c < width - 1: neighbors.append((r, c + 1))
        # Add 8-connectivity if needed
        if connectivity == 8:
            if r > 0 and c > 0: neighbors.append((r - 1, c - 1))
            if r > 0 and c < width - 1: neighbors.append((r - 1, c + 1))
            if r < height - 1 and c > 0: neighbors.append((r + 1, c - 1))
            if r < height - 1 and c < width - 1: neighbors.append((r + 1, c + 1))
        return neighbors

    def add_wall_constraints(self, variables, initial_grid):
        """壁の制約 (Requirement 6.1)
           - Wall cells should align with initial_grid where possible (part of repair objective)
           - Wall continuity: Walls are 0, 2, or 4 wall neighbors.
        """
        print("Adding wall constraints...")
        height, width = initial_grid.shape
        wall_vars = variables['wall']

        for r in range(height):
            for c in range(width):
                # Wall continuity (as per requirement, but this is a topological constraint, not a placement one)
                # This constraint is usually for ensuring walls form valid structures, not for initial placement.
                # Let's assume this constraint applies to the *final* wall configuration.
                neighbor_wall_vars = [wall_vars[nr][nc] for nr, nc in self.get_neighbors(r, c, height, width)]
                sum_neighbor_walls = sum(neighbor_wall_vars)
                
                # If a cell IS a wall, it must have 0, 2, 3 or 4 wall neighbors (common for pixel grids, 3 for T-junctions).
                # The requirement says "0, 2, or 4", which is for non-endpoints of simple paths/loops.
                # This might be too restrictive. Let's follow the requirement for now.
                # A wall cell (wall_vars[r][c] == 1) implies sum_neighbor_walls is in {0, 2, 4}
                # This is tricky: if wall_vars[r][c] is 0, this constraint shouldn't apply.
                # This should be: (wall_vars[r][c] == 1) => (sum_neighbor_walls == 0 OR sum_neighbor_walls == 2 OR sum_neighbor_walls == 4)
                # This can be written using reification or by adding constraints only if wall_vars[r][c] is true.
                
                # Let's use AddAllowedAssignments: (sum_neighbor_walls, wall_vars[r][c])
                # If wall_vars[r][c] == 0 (not a wall), sum_neighbor_walls can be anything.
                # If wall_vars[r][c] == 1 (is a wall), sum_neighbor_walls must be in {0,2,4}.
                # This is not quite right. The constraint should be ON sum_neighbor_walls IF wall_vars[r][c] is 1.

                # Constraint: if wall_vars[r][c] is true, then sum(neighbors) must be in {0,2,4}
                # We need to handle this constraint differently since sum_neighbor_walls is an expression
                # Create intermediate variable for the sum
                sum_var = self.model.NewIntVar(0, 4, f'sum_neighbors_{r}_{c}')
                self.model.Add(sum_var == sum(neighbor_wall_vars))
                
                # Create boolean variables for each allowed value
                is_zero = self.model.NewBoolVar(f'is_zero_{r}_{c}')
                is_two = self.model.NewBoolVar(f'is_two_{r}_{c}')
                is_four = self.model.NewBoolVar(f'is_four_{r}_{c}')
                
                self.model.Add(sum_var == 0).OnlyEnforceIf(is_zero)
                self.model.Add(sum_var != 0).OnlyEnforceIf(is_zero.Not())
                self.model.Add(sum_var == 2).OnlyEnforceIf(is_two)
                self.model.Add(sum_var != 2).OnlyEnforceIf(is_two.Not())
                self.model.Add(sum_var == 4).OnlyEnforceIf(is_four)
                self.model.Add(sum_var != 4).OnlyEnforceIf(is_four.Not())
                
                # At least one must be true if wall_vars[r][c] is true
                self.model.AddBoolOr(is_zero, is_two, is_four).OnlyEnforceIf(wall_vars[r][c])
        print("Wall constraints added.")

    def add_room_constraints(self, variables, initial_grid):
        """部屋の制約 (Requirement 6.1)
           - Min area (6sqm = ~7-8 grids of 0.91x0.91)
           - Rooms are contiguous areas not occupied by walls.
           - Rooms are identified by IDs (1-9).
        """
        print("Adding room constraints...")
        height, width = initial_grid.shape
        room_vars = variables['rooms']
        wall_vars = variables['wall']

        # Constraint: If a cell is a room (room_vars[r][c] > 0), it cannot be a wall.
        for r in range(height):
            for c in range(width):
                is_room_cell = self.model.NewBoolVar(f'is_room_cell_{r}_{c}')
                self.model.Add(room_vars[r][c] > 0).OnlyEnforceIf(is_room_cell)
                self.model.Add(room_vars[r][c] == 0).OnlyEnforceIf(is_room_cell.Not())
                self.model.AddImplication(is_room_cell, wall_vars[r][c].Not())

        min_area_sqm = 6
        # Assuming grid cell is approx 0.91m * 0.91m = 0.8281 sqm
        min_area_grids = int(np.ceil(min_area_sqm / (0.91 * 0.91))) # Approx 8 grids

        for room_id in range(1, 10): # Room IDs 1 through 9
            room_cells_for_id = []
            for r in range(height):
                for c in range(width):
                    is_cell_in_room_id = self.model.NewBoolVar(f'is_cell_in_room_{room_id}_{r}_{c}')
                    self.model.Add(room_vars[r][c] == room_id).OnlyEnforceIf(is_cell_in_room_id)
                    self.model.Add(room_vars[r][c] != room_id).OnlyEnforceIf(is_cell_in_room_id.Not())
                    room_cells_for_id.append(is_cell_in_room_id)
            
            # If room_id exists, its area must be >= min_area_grids
            room_exists = self.model.NewBoolVar(f'room_exists_{room_id}')
            self.model.Add(sum(room_cells_for_id) > 0).OnlyEnforceIf(room_exists) # Room exists if any cell has this ID
            self.model.Add(sum(room_cells_for_id) == 0).OnlyEnforceIf(room_exists.Not())

            self.model.Add(sum(room_cells_for_id) >= min_area_grids).OnlyEnforceIf(room_exists)
            
            # Add contiguity constraint for rooms (complex, often handled by graph-based constraints or specialized methods)
            # For MVP, we might simplify this or rely on the AI to generate mostly contiguous rooms.
            # A full CP-SAT contiguity constraint is non-trivial.
            # Placeholder: self.add_contiguity_for_region(room_vars, room_id, height, width, model)
        print("Room constraints added.")


    def add_connectivity_constraints(self, variables, initial_grid):
        """動線・接続性制約 (Requirement 6.1)
           - All rooms must be reachable from each other (perhaps via other rooms/hallways).
           - Stairs must be reachable.
           This is a global property and hard to enforce directly with local CP-SAT constraints without graph variables.
           Often handled by ensuring doors exist between rooms or rooms and hallways, and hallways connect.
           For an MVP, this might be checked post-generation or simplified.
        """
        print("Placeholder: Adding connectivity constraints...")
        # This is a complex constraint. For MVP, we might rely on post-processing or a simplified version.
        # True connectivity often requires pathfinding or graph-based variables in the CP model.
        pass

    def add_stair_constraints(self, variables, initial_grid):
        """階段の制約 (Requirement 6.1)
           - 1F and 2F stairs align.
           - Stairs form a contiguous block of 4-12 grids.
           - Stairs are not walls.
        """
        print("Adding stair constraints...")
        height, width = initial_grid.shape
        stair_1f_vars = variables['stairs_1f']
        stair_2f_vars = variables['stairs_2f'] # Assuming a 2-story house context
        wall_vars = variables['wall']

        stair_cells_1f_list = []
        for r in range(height):
            for c in range(width):
                # Stairs align
                self.model.Add(stair_1f_vars[r][c] == stair_2f_vars[r][c])
                
                # If a cell is a stair, it cannot be a wall
                self.model.AddImplication(stair_1f_vars[r][c], wall_vars[r][c].Not())
                
                stair_cells_1f_list.append(stair_1f_vars[r][c])
        
        # Stair area constraint (total number of stair cells for 1F)
        total_stair_cells = sum(stair_cells_1f_list)
        self.model.Add(total_stair_cells >= 4)
        self.model.Add(total_stair_cells <= 12)

        # Stair contiguity (complex constraint, similar to room contiguity)
        # Placeholder: self.add_contiguity_for_region(stair_1f_vars, 1, height, width, model)
        # For MVP, might assume AI generates somewhat contiguous stairs or check post-hoc.
        print("Stair constraints added.")

    def add_repair_variables(self, variables, initial_grid):
        """目的関数（最小変更）のための修復変数を追加 (Requirement 6.1)
           Minimize changes from initial_grid for walls.
           Potentially for other elements if they are also part of initial_grid.
        """
        print("Adding repair variables for objective function...")
        height, width = initial_grid.shape
        wall_vars = variables['wall']
        repair_vars_list = []

        for r in range(height):
            for c in range(width):
                # For wall cells
                initial_is_wall = initial_grid[r, c] == 1
                # repair_wall[r][c] is true if wall_vars[r][c] is different from initial_grid[r,c]
                repair_wall_cell = self.model.NewBoolVar(f'repair_wall_{r}_{c}')
                
                if initial_is_wall:
                    # If initial_grid says it's a wall, we want wall_vars[r][c] to be true.
                    # repair_wall_cell is true if wall_vars[r][c] is false.
                    self.model.Add(wall_vars[r][c] == False).OnlyEnforceIf(repair_wall_cell)
                    self.model.Add(wall_vars[r][c] == True).OnlyEnforceIf(repair_wall_cell.Not())
                else:
                    # If initial_grid says it's NOT a wall, we want wall_vars[r][c] to be false.
                    # repair_wall_cell is true if wall_vars[r][c] is true.
                    self.model.Add(wall_vars[r][c] == True).OnlyEnforceIf(repair_wall_cell)
                    self.model.Add(wall_vars[r][c] == False).OnlyEnforceIf(repair_wall_cell.Not())
                repair_vars_list.append(repair_wall_cell)

        # Similar repair variables can be added for rooms, stairs if initial states are given for them.
        print(f"Added {len(repair_vars_list)} repair variables.")
        return repair_vars_list

    def extract_solution(self, variables, height, width):
        """求解結果から修復された平面図データを抽出"""
        print("Extracting solution...")
        # Create an output structure, e.g., a multi-channel image or dict of grids
        # For now, let's return a dictionary of grids
        solution = {
            'walls': np.zeros((height, width), dtype=int),
            'rooms': np.zeros((height, width), dtype=int),
            'stairs_1f': np.zeros((height, width), dtype=int),
            # 'stairs_2f' would be same as 1f based on constraints
        }

        for r in range(height):
            for c in range(width):
                if self.solver.Value(variables['wall'][r][c]):
                    solution['walls'][r,c] = 1
                solution['rooms'][r,c] = self.solver.Value(variables['rooms'][r][c])
                if self.solver.Value(variables['stairs_1f'][r][c]):
                    solution['stairs_1f'][r,c] = 1
        
        print("Solution extracted.")
        # This solution (grids) would then be converted back to an image format if needed by subsequent steps.
        return solution

    def validate_and_fix(self, floor_plan_image, timeout_sec=30):
        """平面図の制約チェックと最小修復 (Requirement 6.1)
           floor_plan_image is the output from the AI model.
           
           Args:
               floor_plan_image: The image output from the AI model (numpy array)
               timeout_sec: Maximum time in seconds to spend on constraint solving
               
           Returns:
               dict: Solution grids if constraints are satisfied, None otherwise
        """
        try:
            print("Starting validation and fix process...")
            self.model = cp_model.CpModel() # Re-initialize model for each call
            self.solver = cp_model.CpSolver()
            
            self.solver.parameters.max_time_in_seconds = timeout_sec
            self.solver.parameters.log_search_progress = True
            
            # 1. グリッド化 (Convert AI output image to a usable grid format)
            try:
                initial_grid = self.image_to_grid(floor_plan_image)
                height, width = initial_grid.shape
                print(f"Initial grid size: {height}x{width}")
            except Exception as e:
                print(f"Error converting image to grid: {e}")
                return None
            
            # 2. 変数定義
            try:
                variables = self.define_variables(height, width)
            except Exception as e:
                print(f"Error defining variables: {e}")
                return None
            
            # 3. 制約定義
            try:
                self.add_wall_constraints(variables, initial_grid)
                self.add_room_constraints(variables, initial_grid)
                self.add_connectivity_constraints(variables, initial_grid)
                self.add_stair_constraints(variables, initial_grid)
            except Exception as e:
                print(f"Error adding constraints: {e}")
                return None
            
            # 4. 目的関数（最小変更）
            try:
                repair_objective_vars = self.add_repair_variables(variables, initial_grid)
                self.model.Minimize(sum(repair_objective_vars))
                print("Objective function set to minimize repairs.")
            except Exception as e:
                print(f"Error setting objective function: {e}")
                return None
            
            # 5. 求解
            print(f"Solving CP-SAT model (timeout: {timeout_sec}s)...")
            status = self.solver.Solve(self.model)
            print(f"Solver status: {self.solver.StatusName(status)}")
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print("Optimal or feasible solution found.")
                solution = self.extract_solution(variables, height, width)
                
                solution['original_grid'] = initial_grid
                
                wall_changes = np.sum(solution['walls'] != initial_grid)
                print(f"Wall cells modified: {wall_changes} out of {height*width}")
                
                return solution
            else:
                print("No solution found or problem was infeasible/unbounded.")
                return None
                
        except Exception as e:
            print(f"Unexpected error in constraint validation: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def visualize_solution(self, solution):
        """制約解決結果の可視化（デバッグ用）
        
        Args:
            solution: The solution dictionary returned by validate_and_fix
            
        Returns:
            numpy.ndarray: Visualization image
        """
        if solution is None:
            print("No solution to visualize")
            return None
            
        # Create a visualization image
        height, width = solution['walls'].shape
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        vis_image[:, :, 2] = solution['walls'] * 255
        
        room_display = np.zeros_like(solution['rooms'])
        for room_id in range(1, 10):  # Room IDs 1-9
            room_display[solution['rooms'] == room_id] = 100 + (room_id * 15)
        vis_image[:, :, 1] = room_display
        
        vis_image[:, :, 0] = solution['stairs_1f'] * 255
        
        vis_image[0, :, :] = 255
        vis_image[-1, :, :] = 255
        vis_image[:, 0, :] = 255
        vis_image[:, -1, :] = 255
        
        return vis_image

# Example Usage (for testing, can be removed or put in a script)
# if __name__ == '__main__':
#     constraint_checker = ArchitecturalConstraints()
#     # Create a dummy floor plan image (e.g., output from AI)
#     # This should ideally be a 4-channel image as per TrainingDataGenerator output format, or a simplified wall grid
#     dummy_ai_output_walls = np.zeros((32, 32), dtype=int) # Example size
#     dummy_ai_output_walls[5:10, 5:25] = 1 # Some horizontal wall
#     dummy_ai_output_walls[5:25, 5:10] = 1 # Some vertical wall
#     dummy_ai_output_walls[15:20, 15:20] = 1 # A small room block
    
#     # If image_to_grid expects multi-channel, adapt dummy_ai_output
#     # For now, let's assume image_to_grid handles a 2D wall grid directly
#     print("Running dummy validation...")
#     solution = constraint_checker.validate_and_fix(dummy_ai_output_walls)

#     if solution:
#         print("Solution Walls:")
#         print(solution['walls'])
#         print("Solution Rooms:")
#         print(solution['rooms'])
#         print("Solution Stairs:")
#         print(solution['stairs_1f'])
#     else:
#         print("No solution could be found for the dummy input.")  