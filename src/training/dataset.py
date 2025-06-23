import torch
from torch.utils.data import Dataset
import json
import os
from glob import glob
import cv2
import numpy as np
import re
import shutil
from pathlib import Path

class FloorPlanDataset(Dataset):

    data_dir : str
    transform : callable
    train_data : list[dict]
    organize_raw : bool
    target_sizes : tuple[int, int]
    channel_count : int

    def __init__(self, data_dir="data", transform=None, organize_raw=False, target_size=512):
        """
        Args:
            data_dir: データディレクトリのパス
            transform: データ変換用の関数（オプション）
            organize_raw: 訓練データの整理を行うかどうか
            target_size: 画像のサイズ
        """
        self.data_dir = data_dir
        self.organize_raw = organize_raw
        self.transform = transform
        self.target_sizes = (target_size, target_size)
        self.channel_count = 4

        self.train_data = self.load_train_data(organize_raw)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if idx >= len(self.train_data):
            raise IndexError("Dataset index out of range")
            
        train_datum = self.train_data[idx]
        dir_path = train_datum['dir_path']
        metadata = train_datum['metadata']
        # 画像読み込み
        img_mask = train_datum['img_mask']
        img_plan = train_datum['img_plan']

        if img_mask is None or img_plan is None:
            print(f"Error loading images for pair {dir_path}. Returning None.")
            return None

        # 正規化 (0-1)
        img_mask = img_mask.astype(np.float32) / 255.0
        img_plan = img_plan.astype(np.float32) / 255.0

        # 4チャンネルのテンソルを作成
        # [walls, openings, stairs, rooms] の形式
        channels = np.zeros((self.channel_count, *self.target_sizes), dtype=np.float32)

        # 現在の3チャンネルデータを4チャンネルにマッピング
        # チャンネル0: stairs（階段）の情報
        channels[0] = img_plan[:, :, 0]  # stairsチャンネル

        # チャンネル1: entrance（玄関）の情報
        channels[1] = img_plan[:, :, 1]  # entranceチャンネル
        
        # チャンネル2: balcony（バルコニー）の情報
        channels[2] = img_plan[:, :, 2]  # balconyチャンネル

        # チャンネル3: 空のチャンネル（必要に応じて後で実装）
        channels[3] = np.zeros(self.target_sizes, dtype=np.float32)

        # PyTorchテンソルに変換
        mask_tensor = torch.from_numpy(img_mask).unsqueeze(0)  # [1, H, W]
        plan_tensor = torch.from_numpy(channels)  # [4, H, W]

        # プロンプト生成
        prompt = self.create_prompt(metadata)

        return {
            'condition': mask_tensor,
            'target': plan_tensor,
            'prompt': prompt,
            'metadata': metadata
        }        

    def extract_floor_num(self, filename):
        """
        ファイル名から数値と階数を抽出する
        
        Args:
            filename: ファイル名（拡張子なし）
            
        Returns:
            str: 数値（5桁0埋め）_階数 または None
        """
        # 階数のパターン（1f, 2f）
        floor_pattern = r'([12])f'
        # 数値のパターン（3-5桁）
        number_pattern = r'(\d{3,5})'
        
        # 階数を検索
        floor_match = re.search(floor_pattern, filename.lower())
        if not floor_match:
            return None
            
        floor = f"{floor_match.group(1)}f"
        
        # 数値を検索
        number_match = re.search(number_pattern, filename)
        if not number_match:
            return None
            
        number = number_match.group(1).zfill(5)
        
        return f"{number}_{floor}"
    
    def load_train_data(self, organize_raw=False):
        # 訓練データ用のディレクトリを作成
        training_dir = os.path.join(self.data_dir, "training")
        if not os.path.exists(training_dir):
            os.makedirs(training_dir, exist_ok=True)

        # rawディレクトリからtrainingディレクトリに整理する
        if organize_raw:
            print("Organizing raw to training ...")
            # rawディレクトリ内のPNGファイルを検索
            raw_dir = os.path.join(self.data_dir, "raw")
            if not os.path.exists(raw_dir):
                print(f"Raw directory not found: {raw_dir}")
                return
        
            png_files = glob(os.path.join(raw_dir, "*.png"))
            print(f"Found {len(png_files)} PNG files in {raw_dir}")
            
            organized_count = 0
            for png_path in png_files:
                try:
                    file_stem = Path(png_path).stem
                    floor_num = self.extract_floor_num(file_stem)
                    
                    if not floor_num:
                        print(f"Could not extract floor num from {file_stem}")
                        continue
                    
                    # 対応するJSONファイルのパス
                    integrated_json_path = png_path.replace('.png', '_integrated.json')
                    elements_json_path = png_path.replace('.png', '_elements.json')
                    
                    if not os.path.exists(integrated_json_path):
                        print(f"Integrated JSON not found for {file_stem}")
                        continue
                    
                    # 訓練データ用のディレクトリを作成
                    target_dir = os.path.join(training_dir, floor_num)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # ファイルをコピー
                    shutil.copy2(png_path, os.path.join(target_dir, "img_base.png"))
                    shutil.copy2(integrated_json_path, os.path.join(target_dir, "meta_integrated.json"))
                    
                    if os.path.exists(elements_json_path):
                        shutil.copy2(elements_json_path, os.path.join(target_dir, "meta_elements.json"))
                    
                    organized_count += 1
                    print(f"Organized {file_stem} -> {floor_num}")
                    
                except Exception as e:
                    print(f"Error organizing {png_path}: {e}")
                    continue
            
            print(f"Successfully organized {organized_count} datasets")

        # meta_integrated.json と img_base.png を読み込み、学習用データを生成する
        train_data = []
        for floor_dir in os.listdir(training_dir):
            floor_path = os.path.join(training_dir, floor_dir)
            if not os.path.isdir(floor_path):
                continue
                
            img_base_path = os.path.join(floor_path, "img_base.png")
            meta_integrated_path = os.path.join(floor_path, "meta_integrated.json")
            meta_elements_path = os.path.join(floor_path, "meta_elements.json")
            
            if not (os.path.exists(img_base_path) and os.path.exists(meta_integrated_path)):
                print(f"Warning: Incomplete data in {floor_dir}")
                continue
            
            try:
                with open(meta_integrated_path, 'r') as f:
                    metadata = json.load(f)
                    
                # 学習用画像合成
                img_plan, img_mask = self.render_pair_images(metadata, img_base_path)
                prompt = self.create_prompt(metadata)

                train_data.append({
                    'dir_path': floor_path,
                    'img_plan': img_plan,
                    'img_mask': img_mask,
                    'metadata': metadata,
                    'prompt': prompt
                })

            except Exception as e:
                print(f"Error loading metadata from {meta_integrated_path}: {e}")
                continue

        print(f"Loaded {len(train_data)} valid datasets.")
        return train_data

    def render_pair_images(self, metadata:dict, img_base_path:str):
        """
        '*_integrated.json' metadataから、*_floor_plan.png, *_floor_mask.png, *_floor_conv.png を生成する

        Args:
            metadata: integrated metadata (json)
            img_base_path: base floor image(png) path 
            
        Returns:
            result_pair: success
            None: failure
        """

        try:
            grid_dimensions = metadata.get('grid_dimensions', None)
            if grid_dimensions is None or not isinstance(grid_dimensions, dict) or len(grid_dimensions) < 2:
                grid_dimensions = {'width_grids': 10, 'height_grids': 10}

            scale_info = metadata.get('scale_info', None)
            if scale_info is None or not isinstance(scale_info, dict) or len(scale_info) < 2:
                scale_info = {
                    'drawing_scale': '1:100',
                    'grid_mm': 910,
                    'grid_px': 107.5
                }

            dir_path = os.path.dirname(img_base_path)
            img_base = cv2.imread(img_base_path)
            height_image, width_image = img_base.shape[:2]

            width_grids = grid_dimensions['width_grids']
            height_grids = grid_dimensions['height_grids']
            width_per_grid = width_image / width_grids
            height_per_grid = height_image / height_grids


            structural_elements = metadata.get('structural_elements', None)
            if structural_elements is None or not isinstance(structural_elements, list):
                structural_elements = [
                    {"type": "stair", "grid_x": 1.0, "grid_y": 1.0, "grid_width": 2.0, "grid_height": 1.0, "name": "stair_1"},
                    {"type": "entrance", "grid_x": 8.0, "grid_y": 8.0, "grid_width": 2.0, "grid_height": 2.0, "name": "entrance_2"},
                    {"type": "balcony", "grid_x": 0.0, "grid_y": 7.0, "grid_width": 3.0, "grid_height": 3.0, "name": "balcony_3"}
                ]
        
            img_base_height = img_base.shape[0]
            img_base_width = img_base.shape[1]

            # tmp_planは、img_baseの上に枠を描画した画像
            tmp_plan = img_base.copy()
            # tmp_maskは、白背景の上に塗りつぶし領域を描画した画像
            tmp_mask = np.ones((img_base_height, img_base_width, 3), dtype=np.uint8) * 255
            for item in structural_elements:
                element_type = item['type']
                grid_x1 = round(item['grid_x'] * width_per_grid)
                grid_y1 = round(item['grid_y'] * height_per_grid)
                grid_x2 = round((item['grid_x'] + item['grid_width']) * width_per_grid)
                grid_y2 = round((item['grid_y'] + item['grid_height']) * height_per_grid)
                fill_color_dict = { "stair": (255, 0, 0), "entrance": (0, 255, 0), "balcony": (0, 0, 255) }
                fill_color = fill_color_dict.get(element_type, (0, 0, 0))
                cv2.rectangle(tmp_plan, (grid_x1, grid_y1), (grid_x2, grid_y2), fill_color, thickness=5)
                cv2.rectangle(tmp_mask, (grid_x1, grid_y1), (grid_x2, grid_y2), fill_color, thickness=-1)

            # img_baseとgrid_dimensionsに合わせて、サイズ・配置を決定
            img_mask_width = int(img_base_width / width_grids * 16)
            img_mask_height = int(img_base_height / height_grids * 16)
            img_mask_size = max(img_mask_width, img_mask_height)
            mask_start_x = int((img_mask_size - img_base_width) / 2)
            mask_start_y = int((img_mask_size - img_base_height) / 2)
            mask_end_x = int(mask_start_x + img_base_width)
            mask_end_y = int(mask_start_y + img_base_height)

            # img_plan : 16x16gridに、img_baseを配置
            img_plan = np.zeros((img_mask_size, img_mask_size, 3), dtype=np.uint8)
            img_plan[mask_start_y:mask_end_y, mask_start_x:mask_end_x] = img_base

            # img_mask : 16x16gridに、階段・玄関・バルコニーのマスクを配置
            img_mask = np.zeros((img_mask_size, img_mask_size, 3), dtype=np.uint8)
            img_mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x] = tmp_mask

            # img_conv : 16x16gridに、img_baseを配置し、階段・玄関・バルコニーの枠を上乗せ
            img_conv = np.zeros((img_mask_size, img_mask_size, 3), dtype=np.uint8)
            img_conv = cv2.rectangle(img_conv, (mask_start_x, mask_start_y), (mask_end_x, mask_end_y), (255, 255, 255), thickness=-1)
            img_conv[mask_start_y:mask_end_y, mask_start_x:mask_end_x] = tmp_plan

            # OpenCVのBGR配色を、PyTorchのRGB配色に変換する
            img_plan = cv2.cvtColor(img_plan, cv2.COLOR_BGR2RGB)
            img_conv = cv2.cvtColor(img_conv, cv2.COLOR_BGR2RGB)
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)

            img_plan = cv2.resize(img_plan, self.target_sizes)
            img_conv = cv2.resize(img_conv, self.target_sizes)
            img_mask = cv2.resize(img_mask, self.target_sizes)

            cv2.imwrite(f"{dir_path}/floor_plan.png", img_plan)
            cv2.imwrite(f"{dir_path}/floor_mask.png", img_mask)
            cv2.imwrite(f"{dir_path}/floor_conv.png", img_conv)

            return img_plan, img_mask

        except Exception as e:
            print(f"Error generate train images from metadata: {e}")

            return None

    def create_prompt(self, metadata):
        """メタデータからプロンプト生成"""
        # Example prompt generation based on metadata with robust error handling
        try:
            # パラメータ取得
            annotation_metadata = metadata.get('annotation_metadata', None)
            if annotation_metadata is None or not isinstance(annotation_metadata, dict) or len(annotation_metadata) < 2:
                annotation_metadata = {
                    'annotator_version': 'v2.0_structural_focus',
                    'annotation_time': 'unknown',
                    'element_count': {
                        'total_elements': 0,
                        'stair_count': 0,
                        'entrance_count': 0,
                        'balcony_count': 0
                    },
                    'floor_type': '1F',
                    'grid_resolution': '6x10',
                    'drawing_scale': '1:100'
                }

            training_hints = metadata.get('training_hints', None)
            if training_hints is None or not isinstance(training_hints, dict) or len(training_hints) < 2:
                training_hints = {
                    'total_area_grids': 60,
                    'room_count': 3,
                    'has_entrance': True,
                    'has_stair': True,
                    'has_balcony': False
                }
            
            original_pdf = metadata.get('original_pdf', 'unknown')
            if original_pdf is None:
                original_pdf = 'unknown'

            grid_dimensions = metadata.get('grid_dimensions', None)
            if grid_dimensions is None or not isinstance(grid_dimensions, dict) or len(grid_dimensions) < 2:
                grid_dimensions = {
                    'width_grids': 10,
                    'height_grids': 10
                }
            
            scale_info = metadata.get('scale_info', None)
            if scale_info is None or not isinstance(scale_info, dict) or len(scale_info) < 2:
                scale_info = {
                    'drawing_scale': '1:100',
                    'grid_mm': 910,
                    'grid_px': 107.5
                }

            floor = metadata.get('floor', None)
            if floor is None:
                floor = '1F'

            building_context = metadata.get('building_context', None)
            if building_context is None or not isinstance(building_context, dict) or len(building_context) < 2:
                building_context = {
                    'type': 'single_family_house',
                    'floors_total': 2,
                    'current_floor': floor,
                    'typical_patterns': {
                        '1F': ['entrance_area', 'stair', 'public_living_space', 'wet_areas', 'storage_zones'],
                        '2F': ['stair', 'private_sleeping_areas', 'work_space', 'utility_area', 'balcony']
                    },
                    'stair_patterns': {
                        'vertical_alignment': 'critical',
                        'u_turn_benefit': 'Space efficiency on 1F for toilet/storage',
                        'size_variation': '1F can have half-width stairs in U-turn configuration'
                    }
                }

            zones = metadata.get('zones', None)
            if zones is None or not isinstance(zones, list) or len(zones) < 2:
                zones = [
                    {'type': 'living', 'approximate_grids': 17, 'priority': 1},
                    {'type': 'private', 'approximate_grids': 14, 'priority': 2},
                    {'type': 'service', 'approximate_grids': 7, 'priority': 3},
                ]

            # プロンプト生成
            # prompt_parts = []
            # prompt_parts.append(f"grid_{grid_dimensions['width_grids']}x{grid_dimensions['height_grids']}")
            # prompt_parts.append(f"building_{building_context['type']}_{building_context['floors_total']}floors")
            # prompt_parts.append(f"current_floor_{floor}")
            # prompt_parts.append("style_modern")

            # if "entrance" in training_hints['floor_constraints']['prohibited_elements']:
            #     prompt_parts.append("entrance_prohibited")
            # elif "entrance" in training_hints['floor_constraints']['required_elements'] or training_hints['has_entrance']:
            #     prompt_parts.append("entrance_required")

            # if "stair" in training_hints['floor_constraints']['prohibited_elements']:
            #     prompt_parts.append("stair_prohibited")
            # elif "stair" in training_hints['floor_constraints']['required_elements'] or training_hints['has_stair']:
            #     prompt_parts.append("stair_required")

            # if "balcony" in training_hints['floor_constraints']['prohibited_elements']:
            #     prompt_parts.append("balcony_prohibited")
            # elif "balcony" in training_hints['floor_constraints']['required_elements'] or training_hints['has_balcony']:
            #     prompt_parts.append("balcony_required")

            # if 'living' in zones:
            #     idx_living = zones.index(next(z for z in zones if z['type'] == 'living'))
            #     prompt_parts.append(f"living_{zones[idx_living]['approximate_grids']}grids")
            # if 'private' in zones:
            #     idx_private = zones.index(next(z for z in zones if z['type'] == 'private'))
            #     prompt_parts.append(f"private_{zones[idx_private]['approximate_grids']}grids")
            # if 'service' in zones:
            #     idx_service = zones.index(next(z for z in zones if z['type'] == 'service'))
            #     prompt_parts.append(f"service_{zones[idx_service]['approximate_grids']}grids")

            # prompt_parts.append("japanese_house")
            # prompt_parts.append(f"{scale_info['grid_mm']}mm_grid")
            # prompt_parts.append("architectural_plan")

            # prompt = ", ".join(prompt_parts)
            prompt = f"architectural plan of a Japanese house, <{grid_dimensions['height_grids']} grids high>, <{grid_dimensions['width_grids']} grids wide>, <{floor} floor>"
            return prompt
            
        except Exception as e:
            print(f"Error generating prompt from metadata: {e}")
            print(f"Using default prompt instead. Metadata: {metadata}")
            return "japanese_house, 910mm_grid, architectural_plan, modern_style"

# Example of a simple transform (optional)
# from torchvision import transforms
# class ToTensorAndNormalize:
#      def __call__(self, image, mask):
#          # Convert numpy arrays to tensors
#          image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#          mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
#          # Apply normalization if needed
#          # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Example for 3 channel image
#          # image_tensor = normalize(image_tensor)
#          return image_tensor, mask_tensor   