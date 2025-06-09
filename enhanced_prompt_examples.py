#!/usr/bin/env python3
"""
Enhanced prompt generation examples for floor plan generation accuracy improvement
Based on plan_118_1f_integrated.json analysis
"""

import json
import os

def generate_basic_structured_prompt(metadata):
    """
    基本構造化プロンプト生成
    統合JSONの主要情報を活用
    """
    grid_dims = metadata.get('grid_dimensions', {})
    width = grid_dims.get('width_grids', 6)
    height = grid_dims.get('height_grids', 10)
    
    elements = []
    for elem in metadata.get('structural_elements', []):
        elem_type = elem['type']
        x = elem['grid_x']
        y = elem['grid_y']
        elements.append(f"{elem_type}_{x}_{y}")
    
    total_grids = metadata.get('training_hints', {}).get('total_area_grids', width * height)
    room_count = metadata.get('training_hints', {}).get('room_count', 3)
    floor = metadata.get('floor', '1F')
    scale = metadata.get('scale_info', {}).get('drawing_scale', '1:100')
    
    elements_str = "_".join(elements) if elements else "no_structural"
    
    prompt = f"grid_{width}x{height}, {elements_str}, total_area_{total_grids}grids, room_count_{room_count}, floor_{floor}, scale_{scale.replace(':', 'to')}, japanese_residential, 910mm_grid"
    
    return prompt

def generate_extended_contextual_prompt(metadata):
    """
    拡張版コンテキスト重視プロンプト
    建築コンテキストとゾーン情報を活用
    """
    building_ctx = metadata.get('building_context', {})
    building_type = building_ctx.get('type', 'single_family_house')
    floors_total = building_ctx.get('floors_total', 2)
    current_floor = building_ctx.get('current_floor', '1F')
    
    grid_dims = metadata.get('grid_dimensions', {})
    grid_info = f"grid_dimensions_{grid_dims.get('width_grids', 6)}x{grid_dims.get('height_grids', 10)}"
    
    structural_parts = []
    for elem in metadata.get('structural_elements', []):
        elem_detail = f"{elem['type']}_grid{elem['grid_x']}x{elem['grid_y']}"
        structural_parts.append(elem_detail)
    structural_str = "_".join(structural_parts) if structural_parts else "no_structural"
    
    zones_info = []
    for zone in metadata.get('zones', []):
        zone_detail = f"{zone['type']}{zone['approximate_grids']}grids"
        zones_info.append(zone_detail)
    zones_str = "_".join(zones_info) if zones_info else "mixed_zones"
    
    stair_info = metadata.get('building_context', {}).get('stair_patterns', {})
    stair_constraint = "stair_alignment_critical" if stair_info.get('vertical_alignment') == 'critical' else "stair_flexible"
    
    scale_info = metadata.get('scale_info', {})
    drawing_scale = scale_info.get('drawing_scale', '1:100').replace(':', 'to')
    
    prompt = f"building_{building_type}_{floors_total}floors, current_floor_{current_floor}, {grid_info}, structural_elements_{structural_str}, zones_{zones_str}, {stair_constraint}, drawing_scale_{drawing_scale}, japanese_residential_910mm_grid, architectural_plan"
    
    return prompt

def generate_hierarchical_conditioning_prompt(metadata):
    """
    階層的条件付けプロンプト
    フロア特性と機能配置を重視
    """
    building_ctx = metadata.get('building_context', {})
    current_floor = building_ctx.get('current_floor', '1F')
    typical_patterns = building_ctx.get('typical_patterns', {})
    floor_functions = typical_patterns.get(current_floor, [])
    
    grid_dims = metadata.get('grid_dimensions', {})
    width = grid_dims.get('width_grids', 6)
    height = grid_dims.get('height_grids', 10)
    total_grids = width * height
    
    stair_details = []
    entrance_details = []
    
    for elem in metadata.get('structural_elements', []):
        if elem['type'] == 'stair':
            stair_details.append(f"stair_u_turn_{elem['grid_width']}x{elem['grid_height']}grids_pos{elem['grid_x']}x{elem['grid_y']}")
        elif elem['type'] == 'entrance':
            entrance_details.append(f"entrance_{elem['grid_width']}x{elem['grid_height']}grids_pos{elem['grid_x']}x{elem['grid_y']}")
    
    functions_str = "+".join(floor_functions) if floor_functions else "mixed_functions"
    stair_str = "_".join(stair_details) if stair_details else "no_stair"
    entrance_str = "_".join(entrance_details) if entrance_details else "no_entrance"
    
    prompt = f"floor_{current_floor}: {functions_str}, grid_{width}x{height}_{total_grids}total, {stair_str}, {entrance_str}, japanese_house_910mm_standard"
    
    return prompt

def generate_constraint_aware_prompt(metadata):
    """
    制約認識プロンプト
    建築制約と検証ルールを重視
    """
    training_hints = metadata.get('training_hints', {})
    
    floor_constraints = training_hints.get('floor_constraints', {})
    required_elements = floor_constraints.get('required_elements', [])
    prohibited_elements = floor_constraints.get('prohibited_elements', [])
    
    has_elements = []
    if training_hints.get('has_entrance', False):
        has_elements.append('entrance')
    if training_hints.get('has_stair', False):
        has_elements.append('stair')
    if training_hints.get('has_balcony', False):
        has_elements.append('balcony')
    
    grid_module = metadata.get('grid_module_info', {})
    base_module = grid_module.get('base_module_mm', 910)
    
    validation = metadata.get('validation_status', {})
    validation_passed = validation.get('passed', True)
    
    required_str = "+".join(required_elements) if required_elements else "no_requirements"
    prohibited_str = "+".join(prohibited_elements) if prohibited_elements else "no_prohibitions"
    elements_str = "+".join(has_elements) if has_elements else "minimal_elements"
    
    prompt = f"constraints_required_{required_str}_prohibited_{prohibited_str}, elements_{elements_str}, module_{base_module}mm, validation_{'passed' if validation_passed else 'failed'}, japanese_architectural_standards"
    
    return prompt

def generate_multi_scale_prompt(metadata):
    """
    マルチスケールプロンプト
    異なる詳細レベルでの条件付け
    """
    grid_dims = metadata.get('grid_dimensions', {})
    basic_info = f"grid_{grid_dims.get('width_grids', 6)}x{grid_dims.get('height_grids', 10)}"
    
    structural_count = len(metadata.get('structural_elements', []))
    element_summary = metadata.get('element_summary', {})
    structure_info = f"elements_{structural_count}_stairs_{element_summary.get('stair_count', 0)}_entrances_{element_summary.get('entrance_count', 0)}"
    
    detailed_positions = []
    for elem in metadata.get('structural_elements', []):
        pos_detail = f"{elem['type']}_at_{elem['grid_x']:.1f}_{elem['grid_y']:.1f}"
        detailed_positions.append(pos_detail)
    detail_info = "_".join(detailed_positions) if detailed_positions else "no_details"
    
    timestamps = metadata.get('timestamps', {})
    annotation_meta = metadata.get('annotation_metadata', {})
    meta_info = f"annotator_{annotation_meta.get('annotator_version', 'unknown')}_resolution_{annotation_meta.get('grid_resolution', '6x10')}"
    
    prompt = f"L1_{basic_info}, L2_{structure_info}, L3_{detail_info}, L4_{meta_info}, japanese_residential_plan"
    
    return prompt

def demonstrate_prompt_generation():
    """
    実際のJSONデータを使用したプロンプト生成デモ
    """
    sample_metadata = {
        "crop_id": "118_2f",
        "floor": "2F",
        "grid_dimensions": {"width_grids": 6, "height_grids": 10},
        "scale_info": {"drawing_scale": "1:100", "grid_mm": 910},
        "building_context": {
            "type": "single_family_house",
            "floors_total": 2,
            "current_floor": "2F",
            "typical_patterns": {
                "1F": ["entrance_area", "stair", "public_living_space", "wet_areas", "storage_zones"],
                "2F": ["stair", "private_sleeping_areas", "work_space", "utility_area", "balcony"]
            },
            "stair_patterns": {"vertical_alignment": "critical"}
        },
        "structural_elements": [
            {"type": "stair", "grid_x": 0.2, "grid_y": 3.5, "grid_width": 1.6, "grid_height": 0.7},
            {"type": "entrance", "grid_x": 3.7, "grid_y": 8.3, "grid_width": 1.3, "grid_height": 1.1}
        ],
        "zones": [
            {"type": "living", "approximate_grids": 17, "priority": 1},
            {"type": "private", "approximate_grids": 14, "priority": 2},
            {"type": "service", "approximate_grids": 7, "priority": 3}
        ],
        "training_hints": {
            "total_area_grids": 60,
            "room_count": 3,
            "has_entrance": True,
            "has_stair": True,
            "has_balcony": False,
            "floor_constraints": {
                "required_elements": ["stair"],
                "prohibited_elements": ["entrance"]
            }
        }
    }
    
    print("=== Enhanced Prompt Generation Examples ===\n")
    
    print("1. Basic Structured Prompt:")
    print(generate_basic_structured_prompt(sample_metadata))
    print()
    
    print("2. Extended Contextual Prompt:")
    print(generate_extended_contextual_prompt(sample_metadata))
    print()
    
    print("3. Hierarchical Conditioning Prompt:")
    print(generate_hierarchical_conditioning_prompt(sample_metadata))
    print()
    
    print("4. Constraint-Aware Prompt:")
    print(generate_constraint_aware_prompt(sample_metadata))
    print()
    
    print("5. Multi-Scale Prompt:")
    print(generate_multi_scale_prompt(sample_metadata))
    print()

if __name__ == "__main__":
    demonstrate_prompt_generation()
