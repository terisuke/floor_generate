"""
Simple rule-based floor plan generator for MVP
"""
import random
import uuid
from typing import List, Tuple, Dict
from src.models import FloorPlan, Room, Wall, Door, Point
from src.config import (
    ROOM_TYPES, ROOM_SIZE_CONSTRAINTS, 
    DEFAULT_FLOOR_WIDTH, DEFAULT_FLOOR_HEIGHT
)

class SimpleFloorPlanGenerator:
    """Simple rule-based floor plan generator"""
    
    def __init__(self, 
                 floor_width: float = DEFAULT_FLOOR_WIDTH,
                 floor_height: float = DEFAULT_FLOOR_HEIGHT):
        self.floor_width = floor_width
        self.floor_height = floor_height
        self.grid_size = 1.0  # 1 meter grid
        
    def generate(self, 
                 room_requirements: Dict[str, int] = None,
                 name: str = "Generated Floor Plan") -> FloorPlan:
        """
        Generate a simple floor plan
        
        Args:
            room_requirements: Dict mapping room types to counts
            name: Name for the floor plan
            
        Returns:
            Generated FloorPlan object
        """
        if room_requirements is None:
            # Default requirements for MVP
            room_requirements = {
                "office": 3,
                "meeting_room": 2,
                "kitchen": 1,
                "bathroom": 2,
                "reception": 1
            }
        
        floor_plan = FloorPlan(
            id=str(uuid.uuid4()),
            name=name,
            width=self.floor_width,
            height=self.floor_height
        )
        
        # Generate rooms using simple grid-based approach
        self._generate_rooms(floor_plan, room_requirements)
        
        # Generate walls based on room boundaries
        self._generate_walls(floor_plan)
        
        # Add doors between rooms
        self._generate_doors(floor_plan)
        
        return floor_plan
    
    def _generate_rooms(self, floor_plan: FloorPlan, requirements: Dict[str, int]) -> None:
        """Generate rooms using a simple grid subdivision approach"""
        # Calculate total number of rooms
        total_rooms = sum(requirements.values())
        
        # Simple grid subdivision
        grid_cols = int(self.floor_width / 5)  # Approximate 5m wide rooms
        grid_rows = int(self.floor_height / 4)  # Approximate 4m tall rooms
        
        # Create a grid of potential room positions
        grid = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Place rooms in grid
        room_id = 0
        for room_type, count in requirements.items():
            for _ in range(count):
                # Find empty grid cell
                placed = False
                attempts = 0
                while not placed and attempts < 100:
                    row = random.randint(0, grid_rows - 1)
                    col = random.randint(0, grid_cols - 1)
                    
                    if grid[row][col] is None:
                        # Calculate room dimensions
                        min_size, max_size = ROOM_SIZE_CONSTRAINTS.get(
                            room_type, (10, 50)
                        )
                        
                        # Simple rectangular rooms
                        room_width = random.uniform(4, 8)
                        room_height = random.uniform(3, 6)
                        
                        # Calculate corners
                        x = col * 5
                        y = row * 4
                        
                        # Ensure room fits within floor plan
                        if x + room_width <= self.floor_width and y + room_height <= self.floor_height:
                            corners = [
                                Point(x, y),
                                Point(x + room_width, y),
                                Point(x + room_width, y + room_height),
                                Point(x, y + room_height)
                            ]
                            
                            room = Room(
                                id=f"room_{room_id}",
                                type=room_type,
                                corners=corners
                            )
                            
                            floor_plan.add_room(room)
                            grid[row][col] = room_id
                            room_id += 1
                            placed = True
                    
                    attempts += 1
    
    def _generate_walls(self, floor_plan: FloorPlan) -> None:
        """Generate walls based on room boundaries"""
        # Add exterior walls
        floor_plan.add_wall(Wall(Point(0, 0), Point(self.floor_width, 0)))
        floor_plan.add_wall(Wall(Point(self.floor_width, 0), Point(self.floor_width, self.floor_height)))
        floor_plan.add_wall(Wall(Point(self.floor_width, self.floor_height), Point(0, self.floor_height)))
        floor_plan.add_wall(Wall(Point(0, self.floor_height), Point(0, 0)))
        
        # Add walls for each room
        for room in floor_plan.rooms:
            for i in range(len(room.corners)):
                start = room.corners[i]
                end = room.corners[(i + 1) % len(room.corners)]
                
                # Check if this wall already exists (shared wall)
                wall_exists = False
                for wall in floor_plan.walls:
                    if (self._points_equal(wall.start, start) and self._points_equal(wall.end, end)) or \
                       (self._points_equal(wall.start, end) and self._points_equal(wall.end, start)):
                        wall_exists = True
                        break
                
                if not wall_exists:
                    floor_plan.add_wall(Wall(start, end))
    
    def _generate_doors(self, floor_plan: FloorPlan) -> None:
        """Add doors to rooms"""
        for room in floor_plan.rooms:
            # Add at least one door per room
            # For MVP, place door on the first wall
            if len(room.corners) >= 2:
                wall_start = room.corners[0]
                wall_end = room.corners[1]
                
                # Place door in the middle of the wall
                door_x = (wall_start.x + wall_end.x) / 2
                door_y = (wall_start.y + wall_end.y) / 2
                
                door = Door(position=Point(door_x, door_y))
                floor_plan.add_door(door)
    
    def _points_equal(self, p1: Point, p2: Point, tolerance: float = 0.01) -> bool:
        """Check if two points are equal within tolerance"""
        return abs(p1.x - p2.x) < tolerance and abs(p1.y - p2.y) < tolerance


def generate_sample_floor_plan():
    """Generate a sample floor plan for testing"""
    generator = SimpleFloorPlanGenerator()
    return generator.generate()
