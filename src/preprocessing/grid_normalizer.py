"""
Grid Normalizer
Normalize dimensions to 910mm/455mm mixed grid system
"""

import numpy as np
from typing import List, Dict, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridNormalizer:
    """Normalize dimensions to Japanese architectural grid system"""
    
    def __init__(self, primary_grid: int = 910, secondary_grid: int = 455):
        """
        Initialize grid normalizer
        
        Args:
            primary_grid: Primary grid size in mm (default: 910mm - 本間)
            secondary_grid: Secondary grid size in mm (default: 455mm - 半間)
        """
        self.primary = primary_grid     # 910mm (本間)
        self.secondary = secondary_grid # 455mm (半間)
        
        # Common room sizes in grid units
        self.standard_room_sizes = {
            '4.5畳': {'primary': (3, 3), 'area_sqm': 7.29},
            '6畳': {'primary': (3, 4), 'area_sqm': 9.72},
            '8畳': {'primary': (4, 4), 'area_sqm': 12.96},
            '10畳': {'primary': (4, 5), 'area_sqm': 16.20},
            '12畳': {'primary': (4, 6), 'area_sqm': 19.44},
        }
    
    def normalize_dimensions(self, dimensions: List[Dict]) -> List[Dict]:
        """
        Normalize dimensions to mixed grid system
        
        Args:
            dimensions: List of dimension dictionaries from extractor
            
        Returns:
            List of normalized dimension dictionaries
        """
        normalized = []
        
        for dim_info in dimensions:
            dim_value = dim_info.get('value')
            
            if not dim_value:
                continue
            
            if isinstance(dim_value, list):
                # Area dimensions (width x height)
                norm_dims = [self.normalize_single(d) for d in dim_value]
                normalized.append({
                    'original': dim_value,
                    'normalized': norm_dims,
                    'type': 'area',
                    'grid_size': [nd['grid_count'] for nd in norm_dims],
                    'confidence': dim_info.get('confidence', 0),
                    'page': dim_info.get('page', 1)
                })
            else:
                # Linear dimension
                norm_dim = self.normalize_single(dim_value)
                normalized.append({
                    'original': dim_value,
                    'normalized_mm': norm_dim['normalized_mm'],
                    'grid_count': norm_dim['grid_count'],
                    'grid_type': norm_dim['grid_type'],
                    'error_percent': norm_dim['error_percent'],
                    'type': 'linear',
                    'confidence': dim_info.get('confidence', 0),
                    'page': dim_info.get('page', 1)
                })
        
        return normalized
    
    def normalize_single(self, dimension: float) -> Dict:
        """
        Normalize a single dimension value
        
        Args:
            dimension: Dimension value in mm
            
        Returns:
            Dictionary with normalized information
        """
        # Try primary grid (910mm)
        primary_grids = round(dimension / self.primary)
        primary_value = primary_grids * self.primary
        primary_error = abs(dimension - primary_value)
        primary_error_percent = (primary_error / dimension * 100) if dimension > 0 else 0
        
        # Try secondary grid (455mm)
        secondary_grids = round(dimension / self.secondary)
        secondary_value = secondary_grids * self.secondary
        secondary_error = abs(dimension - secondary_value)
        secondary_error_percent = (secondary_error / dimension * 100) if dimension > 0 else 0
        
        # Try mixed grid (combination of primary and secondary)
        mixed_result = self.try_mixed_grid(dimension)
        
        # Choose the best fit (lowest error)
        results = [
            {
                'normalized_mm': primary_value,
                'grid_count': primary_grids,
                'grid_type': 'primary',
                'error_mm': primary_error,
                'error_percent': primary_error_percent
            },
            {
                'normalized_mm': secondary_value,
                'grid_count': secondary_grids,
                'grid_type': 'secondary',
                'error_mm': secondary_error,
                'error_percent': secondary_error_percent
            }
        ]
        
        if mixed_result:
            results.append(mixed_result)
        
        # Return the result with minimum error
        best_result = min(results, key=lambda x: x['error_percent'])
        
        # Log if error is significant
        if best_result['error_percent'] > 5:
            logger.warning(f"High normalization error: {dimension}mm -> "
                         f"{best_result['normalized_mm']}mm "
                         f"({best_result['error_percent']:.1f}% error)")
        
        return best_result
    
    def try_mixed_grid(self, dimension: float) -> Dict:
        """
        Try to fit dimension using combination of primary and secondary grids
        
        Args:
            dimension: Dimension value in mm
            
        Returns:
            Mixed grid result or None
        """
        best_error = float('inf')
        best_combination = None
        
        # Try combinations up to reasonable limits
        max_primary = int(dimension / self.primary) + 2
        max_secondary = int(dimension / self.secondary) + 2
        
        for p in range(max_primary + 1):
            for s in range(max_secondary + 1):
                if p == 0 and s == 0:
                    continue
                    
                total = p * self.primary + s * self.secondary
                error = abs(dimension - total)
                
                if error < best_error:
                    best_error = error
                    best_combination = (p, s)
        
        if best_combination and best_error < dimension * 0.05:  # Less than 5% error
            p, s = best_combination
            total = p * self.primary + s * self.secondary
            
            return {
                'normalized_mm': total,
                'grid_count': f"{p}P+{s}S",  # e.g., "3P+1S" = 3*910 + 1*455
                'grid_type': 'mixed',
                'error_mm': best_error,
                'error_percent': (best_error / dimension * 100) if dimension > 0 else 0,
                'primary_count': p,
                'secondary_count': s
            }
        
        return None
    
    def dimension_to_grid_coords(self, dimension_mm: float, 
                               grid_type: str = 'auto') -> Tuple[int, str]:
        """
        Convert dimension to grid coordinates
        
        Args:
            dimension_mm: Dimension in millimeters
            grid_type: 'primary', 'secondary', or 'auto'
            
        Returns:
            Tuple of (grid_count, actual_grid_type)
        """
        if grid_type == 'auto':
            result = self.normalize_single(dimension_mm)
            return result['grid_count'], result['grid_type']
        elif grid_type == 'primary':
            return round(dimension_mm / self.primary), 'primary'
        elif grid_type == 'secondary':
            return round(dimension_mm / self.secondary), 'secondary'
        else:
            raise ValueError(f"Invalid grid_type: {grid_type}")
    
    def grid_to_pixels(self, grid_count: Union[int, str], 
                      pixels_per_grid: int = 32) -> int:
        """
        Convert grid count to pixels for image generation
        
        Args:
            grid_count: Number of grids or mixed grid string
            pixels_per_grid: Pixels per grid unit
            
        Returns:
            Number of pixels
        """
        if isinstance(grid_count, str) and '+' in grid_count:
            # Mixed grid format (e.g., "3P+1S")
            parts = grid_count.replace('P', '').replace('S', '').split('+')
            primary_count = int(parts[0]) if parts[0] else 0
            secondary_count = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            
            # Convert to primary grid equivalent
            total_primary = primary_count + (secondary_count * self.secondary / self.primary)
            return int(total_primary * pixels_per_grid)
        else:
            return int(grid_count * pixels_per_grid)
    
    def analyze_site_dimensions(self, width_mm: float, depth_mm: float) -> Dict:
        """
        Analyze site dimensions and suggest grid layout
        
        Args:
            width_mm: Site width in mm
            depth_mm: Site depth in mm
            
        Returns:
            Site analysis dictionary
        """
        width_norm = self.normalize_single(width_mm)
        depth_norm = self.normalize_single(depth_mm)
        
        # Calculate area
        area_sqm = (width_mm * depth_mm) / 1_000_000
        normalized_area_sqm = (width_norm['normalized_mm'] * 
                              depth_norm['normalized_mm']) / 1_000_000
        
        # Determine site size category
        if area_sqm < 100:
            size_category = 'small'
        elif area_sqm < 150:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        return {
            'width': {
                'original_mm': width_mm,
                'normalized_mm': width_norm['normalized_mm'],
                'grid_count': width_norm['grid_count'],
                'grid_type': width_norm['grid_type']
            },
            'depth': {
                'original_mm': depth_mm,
                'normalized_mm': depth_norm['normalized_mm'],
                'grid_count': depth_norm['grid_count'],
                'grid_type': depth_norm['grid_type']
            },
            'area_sqm': area_sqm,
            'normalized_area_sqm': normalized_area_sqm,
            'size_category': size_category,
            'recommended_layout': self.suggest_layout(width_norm['grid_count'], 
                                                     depth_norm['grid_count'],
                                                     size_category)
        }
    
    def suggest_layout(self, width_grids: Union[int, str], 
                      depth_grids: Union[int, str], 
                      size_category: str) -> Dict:
        """
        Suggest room layout based on site dimensions
        
        Args:
            width_grids: Width in grid units
            depth_grids: Depth in grid units
            size_category: 'small', 'medium', or 'large'
            
        Returns:
            Layout suggestion dictionary
        """
        # Convert to integers if needed
        w = width_grids if isinstance(width_grids, int) else int(str(width_grids).split('+')[0])
        d = depth_grids if isinstance(depth_grids, int) else int(str(depth_grids).split('+')[0])
        
        suggestions = {
            'small': {
                'rooms': '3LDK',
                'floors': 2,
                'key_features': ['Compact design', 'Efficient circulation']
            },
            'medium': {
                'rooms': '4LDK',
                'floors': 2,
                'key_features': ['Standard family layout', 'Separate living/dining']
            },
            'large': {
                'rooms': '5LDK',
                'floors': 2,
                'key_features': ['Spacious rooms', 'Multiple bathrooms', 'Study room']
            }
        }
        
        return suggestions.get(size_category, suggestions['medium'])


def main():
    """Test the grid normalizer"""
    normalizer = GridNormalizer()
    
    # Test dimensions
    test_dimensions = [
        {'value': 9100},    # Exactly 10 grids
        {'value': 4550},    # Exactly 5 grids  
        {'value': 3640},    # 4 grids
        {'value': 2730},    # 3 grids
        {'value': 5460},    # 6 grids
        {'value': [13650, 10010]},  # Site dimensions
    ]
    
    print("Grid Normalization Test Results:")
    print("=" * 60)
    
    normalized = normalizer.normalize_dimensions(test_dimensions)
    
    for i, norm in enumerate(normalized):
        print(f"\n{i+1}. Original: {norm['original']}mm")
        
        if norm['type'] == 'linear':
            print(f"   Normalized: {norm['normalized_mm']}mm")
            print(f"   Grid: {norm['grid_count']} {norm['grid_type']}")
            print(f"   Error: {norm['error_percent']:.1f}%")
        else:
            print(f"   Type: Area dimensions")
            print(f"   Grid size: {norm['grid_size']}")
            
            # Analyze as site
            if len(norm['original']) == 2:
                analysis = normalizer.analyze_site_dimensions(
                    norm['original'][0], 
                    norm['original'][1]
                )
                print(f"   Area: {analysis['area_sqm']:.1f}㎡")
                print(f"   Category: {analysis['size_category']}")
                print(f"   Suggested: {analysis['recommended_layout']['rooms']}")


if __name__ == "__main__":
    main()
