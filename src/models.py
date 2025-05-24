"""
Data models for floor plan generation
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import numpy as np

@dataclass
class Point:
    """2D point representation"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}

@dataclass
class Room:
    """Room representation"""
    id: str
    type: str
    corners: List[Point]
    center: Optional[Point] = None
    area: Optional[float] = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = self.calculate_center()
        if self.area is None:
            self.area = self.calculate_area()
    
    def calculate_center(self) -> Point:
        """Calculate the center point of the room"""
        if not self.corners:
            return Point(0, 0)
        x_coords = [p.x for p in self.corners]
        y_coords = [p.y for p in self.corners]
        return Point(sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    def calculate_area(self) -> float:
        """Calculate the area of the room using the shoelace formula"""
        if len(self.corners) < 3:
            return 0.0
        
        n = len(self.corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.corners[i].x * self.corners[j].y
            area -= self.corners[j].x * self.corners[i].y
        return abs(area) / 2.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "corners": [p.to_dict() for p in self.corners],
            "center": self.center.to_dict() if self.center else None,
            "area": self.area
        }

@dataclass
class Wall:
    """Wall representation"""
    start: Point
    end: Point
    thickness: float = 0.2  # meters
    
    def length(self) -> float:
        """Calculate wall length"""
        return np.sqrt((self.end.x - self.start.x)**2 + (self.end.y - self.start.y)**2)
    
    def to_dict(self) -> Dict:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "thickness": self.thickness,
            "length": self.length()
        }

@dataclass
class Door:
    """Door representation"""
    position: Point
    width: float = 0.9  # meters
    wall: Optional[Wall] = None
    
    def to_dict(self) -> Dict:
        return {
            "position": self.position.to_dict(),
            "width": self.width
        }

@dataclass
class FloorPlan:
    """Complete floor plan representation"""
    id: str
    name: str
    width: float  # meters
    height: float  # meters
    rooms: List[Room] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_room(self, room: Room) -> None:
        """Add a room to the floor plan"""
        self.rooms.append(room)
    
    def add_wall(self, wall: Wall) -> None:
        """Add a wall to the floor plan"""
        self.walls.append(wall)
    
    def add_door(self, door: Door) -> None:
        """Add a door to the floor plan"""
        self.doors.append(door)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "rooms": [r.to_dict() for r in self.rooms],
            "walls": [w.to_dict() for w in self.walls],
            "doors": [d.to_dict() for d in self.doors],
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FloorPlan':
        """Create FloorPlan from dictionary"""
        floor_plan = cls(
            id=data["id"],
            name=data["name"],
            width=data["width"],
            height=data["height"],
            metadata=data.get("metadata", {})
        )
        
        # Add rooms
        for room_data in data.get("rooms", []):
            corners = [Point(p["x"], p["y"]) for p in room_data["corners"]]
            room = Room(
                id=room_data["id"],
                type=room_data["type"],
                corners=corners
            )
            floor_plan.add_room(room)
        
        # Add walls
        for wall_data in data.get("walls", []):
            wall = Wall(
                start=Point(wall_data["start"]["x"], wall_data["start"]["y"]),
                end=Point(wall_data["end"]["x"], wall_data["end"]["y"]),
                thickness=wall_data.get("thickness", 0.2)
            )
            floor_plan.add_wall(wall)
        
        # Add doors
        for door_data in data.get("doors", []):
            door = Door(
                position=Point(door_data["position"]["x"], door_data["position"]["y"]),
                width=door_data.get("width", 0.9)
            )
            floor_plan.add_door(door)
        
        return floor_plan
