#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import json
import random

def create_floor_plan_data(output_dir, pair_index, width=512, height=512):
    """Create realistic floor plan training data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create site mask (敷地マスク)
    site_mask = np.ones((height, width), dtype=np.uint8) * 255  # 白い背景
    
    # ランダムな敷地サイズ
    grid_size = 32  # 910mmグリッド（512px / 16グリッド）
    site_grids_x = random.randint(8, 14)
    site_grids_y = random.randint(8, 14)
    
    site_width = site_grids_x * grid_size
    site_height = site_grids_y * grid_size
    
    # 敷地を中央に配置
    start_x = (width - site_width) // 2
    start_y = (height - site_height) // 2
    
    # 敷地エリアを黒く塗る
    site_mask[start_y:start_y + site_height, start_x:start_x + site_width] = 0
    
    # Floor plan (間取り図) - RGBA形式
    floor_plan = np.zeros((height, width, 4), dtype=np.uint8)
    floor_plan[:, :, 3] = 255  # Alpha channel
    
    # 壁を描画（赤チャンネル）
    wall_thickness = 3
    
    # 外壁
    floor_plan[start_y:start_y + wall_thickness, start_x:start_x + site_width, 0] = 255
    floor_plan[start_y + site_height - wall_thickness:start_y + site_height, start_x:start_x + site_width, 0] = 255
    floor_plan[start_y:start_y + site_height, start_x:start_x + wall_thickness, 0] = 255
    floor_plan[start_y:start_y + site_height, start_x + site_width - wall_thickness:start_x + site_width, 0] = 255
    
    # 内壁（部屋の区切り）
    num_rooms = random.randint(3, 6)
    
    # 横方向の区切り
    if site_grids_x > 10:
        x_divider = start_x + (site_width // 2)
        floor_plan[start_y:start_y + site_height, x_divider:x_divider + wall_thickness, 0] = 255
    
    # 縦方向の区切り
    if site_grids_y > 10:
        y_divider = start_y + (site_height // 2)
        floor_plan[y_divider:y_divider + wall_thickness, start_x:start_x + site_width, 0] = 255
    
    # 開口部（緑チャンネル）- ドアや窓
    door_width = grid_size // 2
    door_positions = []
    
    # 玄関ドア（下側）
    door_x = start_x + site_width // 2 - door_width // 2
    floor_plan[start_y + site_height - wall_thickness:start_y + site_height, door_x:door_x + door_width, 1] = 255
    floor_plan[start_y + site_height - wall_thickness:start_y + site_height, door_x:door_x + door_width, 0] = 0  # 壁を削除
    
    # 内部ドア
    if 'x_divider' in locals():
        # 横の壁にドア
        door_y = start_y + site_height // 4
        floor_plan[door_y:door_y + door_width, x_divider:x_divider + wall_thickness, 1] = 255
        floor_plan[door_y:door_y + door_width, x_divider:x_divider + wall_thickness, 0] = 0
    
    # 階段（青チャンネル）- ランダムに配置
    if random.random() > 0.5:
        stairs_x = start_x + grid_size
        stairs_y = start_y + grid_size
        stairs_width = grid_size * 2
        stairs_height = grid_size
        floor_plan[stairs_y:stairs_y + stairs_height, stairs_x:stairs_x + stairs_width, 2] = 255
    
    # 部屋（アルファチャンネル）- 部屋の種類を数値で表現
    # 簡単のため、エリアを分割して部屋IDを割り当て
    room_id = 1
    if 'x_divider' in locals() and 'y_divider' in locals():
        # 4分割
        floor_plan[start_y:y_divider, start_x:x_divider, 3] = room_id * 25
        floor_plan[start_y:y_divider, x_divider:start_x + site_width, 3] = (room_id + 1) * 25
        floor_plan[y_divider:start_y + site_height, start_x:x_divider, 3] = (room_id + 2) * 25
        floor_plan[y_divider:start_y + site_height, x_divider:start_x + site_width, 3] = (room_id + 3) * 25
    
    # 画像を保存
    site_mask_img = Image.fromarray(site_mask)
    site_mask_path = os.path.join(output_dir, "site_mask.png")
    site_mask_img.save(site_mask_path)
    
    floor_plan_img = Image.fromarray(floor_plan)
    floor_plan_path = os.path.join(output_dir, "floor_plan.png")
    floor_plan_img.save(floor_plan_path)
    
    # メタデータ
    room_types = ["living", "kitchen", "bedroom", "bathroom", "entrance", "storage"]
    selected_rooms = random.sample(room_types, min(num_rooms, len(room_types)))
    
    metadata = {
        "site_grid_size": [site_grids_x, site_grids_y],
        "total_area_sqm": site_grids_x * site_grids_y * 0.91 * 0.91,
        "room_count": num_rooms,
        "source_pdf": f"synthetic_{pair_index:04d}.pdf",
        "room_types": selected_rooms,
        "floor_count": 1,
        "style": random.choice(["modern", "traditional", "minimalist"])
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created training pair {pair_index:04d}")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create additional training data for floor plan generation")
    parser.add_argument("--output_dir", type=str, default="data/training",
                        help="Output directory for training data")
    parser.add_argument("--num_pairs", type=int, default=10,
                        help="Number of training pairs to create")
    parser.add_argument("--start_index", type=int, default=2,
                        help="Starting index for pair numbering")
    
    args = parser.parse_args()
    
    for i in range(args.num_pairs):
        pair_index = args.start_index + i
        pair_dir = os.path.join(args.output_dir, f"pair_{pair_index:04d}")
        create_floor_plan_data(pair_dir, pair_index)
    
    print(f"\nCreated {args.num_pairs} training pairs in {args.output_dir}")

if __name__ == "__main__":
    main()
