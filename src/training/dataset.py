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
    pairs : list[dict]
    organize_training_data : bool
    target_size : tuple[int, int]
    channel_count : int

    def __init__(self, data_dir, transform=None, organize_training_data=False, target_size=(256, 256)):
        """
        Args:
            data_dir: データディレクトリのパス
            transform: データ変換用の関数（オプション）
            organize_training_data: 訓練データの整理を行うかどうか
            target_size: 画像のサイズ
        """
        self.data_dir = data_dir
        self.transform = transform
        self.organize_training_data = organize_training_data
        self.target_size = target_size
        self.channel_count = 4

        if organize_training_data:
            self.organize_data()
            
        self.pairs = self.load_data_pairs()
        self.pairs = self.generate_train_pairs()

    def extract_floor_info(self, filename):
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

    def organize_data(self):
        """訓練データを整理する"""
        print("Organizing training data...")
        
        # 訓練データ用のディレクトリを作成
        training_dir = os.path.join(self.data_dir, "training")
        os.makedirs(training_dir, exist_ok=True)
        
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
                # ファイル名から情報を抽出
                file_stem = Path(png_path).stem
                floor_info = self.extract_floor_info(file_stem)
                
                if not floor_info:
                    print(f"Could not extract floor info from {file_stem}")
                    continue
                
                # 対応するJSONファイルのパス
                integrated_json_path = png_path.replace('.png', '_integrated.json')
                elements_json_path = png_path.replace('.png', '_elements.json')
                
                if not os.path.exists(integrated_json_path):
                    print(f"Integrated JSON not found for {file_stem}")
                    continue
                
                # 訓練データ用のディレクトリを作成
                target_dir = os.path.join(training_dir, floor_info)
                os.makedirs(target_dir, exist_ok=True)
                
                # ファイルをコピー
                shutil.copy2(png_path, os.path.join(target_dir, "img_base.png"))
                shutil.copy2(integrated_json_path, os.path.join(target_dir, "meta_integrated.json"))
                
                if os.path.exists(elements_json_path):
                    shutil.copy2(elements_json_path, os.path.join(target_dir, "meta_elements.json"))
                
                organized_count += 1
                print(f"Organized {file_stem} -> {floor_info}")
                
            except Exception as e:
                print(f"Error organizing {png_path}: {e}")
                continue
        
        print(f"Successfully organized {organized_count} data pairs")

    def load_data_pairs(self):
        """学習データペアを読み込み"""
        pairs = []
        
        # trainingディレクトリから読み込む
        training_dir = os.path.join(self.data_dir, "training")
        if os.path.exists(training_dir):
            # 訓練データディレクトリ内の各フォルダを処理
            for floor_dir in os.listdir(training_dir):
                floor_path = os.path.join(training_dir, floor_dir)
                if not os.path.isdir(floor_path):
                    continue
                    
                img_base_path = os.path.join(floor_path, "img_base.png")
                meta_integrated_path = os.path.join(floor_path, "meta_integrated.json")
                
                if not (os.path.exists(img_base_path) and os.path.exists(meta_integrated_path)):
                    print(f"Warning: Incomplete data in {floor_dir}")
                    continue
                
                try:
                    with open(meta_integrated_path, 'r') as f:
                        metadata = json.load(f)
                        
                    pairs.append({
                        'dir_path': floor_path,
                        'metadata': metadata,
                        'base_png': img_base_path
                    })
                except Exception as e:
                    print(f"Error loading metadata from {meta_integrated_path}: {e}")
                    continue

        print(f"Loaded {len(pairs)} valid data pairs.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx >= len(self.pairs):
            raise IndexError("Dataset index out of range")
            
        pair = self.pairs[idx]
        dir_path = pair['dir_path']
        metadata = pair['metadata']

        # 画像読み込み
        site_mask = cv2.imread(f"{dir_path}/site_mask.png", cv2.IMREAD_GRAYSCALE)
        floor_plan = cv2.imread(f"{dir_path}/floor_plan.png", cv2.IMREAD_UNCHANGED)

        if site_mask is None or floor_plan is None:
            print(f"Error loading images for pair {dir_path}. Returning None.")
            return None

        # 正規化 (0-1)
        site_mask = site_mask.astype(np.float32) / 255.0
        floor_plan = floor_plan.astype(np.float32) / 255.0

        # 4チャンネルのテンソルを作成
        # [walls, openings, stairs, rooms] の形式
        channels = np.zeros((self.channel_count, *self.target_size), dtype=np.float32)

        # 現在の3チャンネルデータを4チャンネルにマッピング
        # チャンネル0: stairs（階段）の情報
        channels[0] = floor_plan[:, :, 0]  # stairsチャンネル

        # チャンネル1: entrance（玄関）の情報
        channels[1] = floor_plan[:, :, 1]  # entranceチャンネル
        
        # チャンネル2: balcony（バルコニー）の情報
        channels[2] = floor_plan[:, :, 2]  # balconyチャンネル

        # チャンネル3: 空のチャンネル（必要に応じて後で実装）
        channels[3] = np.zeros(self.target_size, dtype=np.float32)

        # PyTorchテンソルに変換
        site_mask_tensor = torch.from_numpy(site_mask).unsqueeze(0)  # [1, H, W]
        floor_plan_tensor = torch.from_numpy(channels)  # [4, H, W]

        # プロンプト生成
        prompt = self.generate_prompt(metadata)

        return {
            'condition': site_mask_tensor,
            'target': floor_plan_tensor,
            'prompt': prompt,
            'metadata': metadata
        }

    def generate_train_pairs(self):
        """
        メタデータをもとに学習用画像データを生成する
        """

        new_pairs = []
        try:
            for pair in self.pairs:
                result_pair = self.generate_train_images(pair['metadata'], pair['base_png'])
                prompt = self.generate_prompt(pair['metadata'])
                result_pair['prompt'] = prompt
                if result_pair is not None:
                    new_pairs.append(result_pair)
                else:
                    print(f"Error generate train images from metadata: {pair['metadata']}")
                    continue

            print(f"Successfully generated {len(new_pairs)} train pairs")
            self.pairs = new_pairs
            return new_pairs

        except Exception as e:
            print(f"Error generate train pairs: {e}")
            return None

    def generate_train_images(self, metadata:dict, img_base_path:str):
        """
        '*_integrated.json' metadataから、256x256pxの *_floor_plan.png, *_site_mask.png, *_conv.png を生成する

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

            width_grids = grid_dimensions['width_grids']
            height_grids = grid_dimensions['height_grids']

            dir_path = os.path.dirname(img_base_path)
            img_base = cv2.imread(img_base_path)
            img_conv = img_base.copy()
            img_conv = cv2.resize(img_conv, (width_grids*10, height_grids*10))

            structural_elements = metadata.get('structural_elements', None)
            if structural_elements is None or not isinstance(structural_elements, list):
                structural_elements = [
                    {"type": "stair", "grid_x": 1.0, "grid_y": 1.0, "grid_width": 2.0, "grid_height": 1.0, "name": "stair_1"},
                    {"type": "entrance", "grid_x": 8.0, "grid_y": 8.0, "grid_width": 2.0, "grid_height": 2.0, "name": "entrance_2"},
                    {"type": "balcony", "grid_x": 0.0, "grid_y": 7.0, "grid_width": 3.0, "grid_height": 3.0, "name": "balcony_3"}
                ]
        
            # 幅高さが小数点1位まであるため、グリッドの10倍で描画
            img_plan = np.zeros((height_grids*10, width_grids*10, 3), dtype=np.uint8)
            for item in structural_elements:
                element_type = item['type']
                grid_x1 = round(item['grid_x'] * 10)
                grid_y1 = round(item['grid_y'] * 10)
                grid_x2 = round((item['grid_x'] + item['grid_width']) * 10)
                grid_y2 = round((item['grid_y'] + item['grid_height']) * 10)
                # 階段は赤、玄関は緑、バルコニーは青、他は黒
                fill_color_dict = { "stair": (255, 0, 0), "entrance": (0, 255, 0), "balcony": (0, 0, 255) }
                fill_color = fill_color_dict.get(element_type, (0, 0, 0))
                cv2.rectangle(img_plan, (grid_x1, grid_y1), (grid_x2, grid_y2), fill_color, thickness=1)
                cv2.rectangle(img_conv, (grid_x1, grid_y1), (grid_x2, grid_y2), fill_color, thickness=1)

            img_plan = cv2.cvtColor(img_plan, cv2.COLOR_RGB2BGR)
            img_conv = cv2.cvtColor(img_conv, cv2.COLOR_RGB2BGR)

            # マスクに外枠をつける
            margin = 0.1
            img_mask = np.zeros((height_grids*10, width_grids*10, 3), dtype=np.uint8)
            img_mask = cv2.rectangle(img_mask, (int(width_grids*10*margin), int(height_grids*10*margin)), (int(width_grids*10*(1-margin)), int(height_grids*10*(1-margin))), (255, 255, 255), thickness=-1)
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

            img_plan = cv2.resize(img_plan, self.target_size)
            img_mask = cv2.resize(img_mask, self.target_size)
            img_conv = cv2.resize(img_conv, self.target_size)

            cv2.imwrite(f"{dir_path}/floor_plan.png", img_plan)
            cv2.imwrite(f"{dir_path}/site_mask.png", img_mask)
            cv2.imwrite(f"{dir_path}/conv.png", img_conv)

            result_pair = {
                "dir_path": dir_path,
                "floor_plan": img_plan,
                "site_mask": img_mask,
                "metadata": metadata
            }
            return result_pair

        except Exception as e:
            print(f"Error generate train images from metadata: {e}")

            return None

    def generate_prompt(self, metadata):
        """メタデータからプロンプト生成"""
        # Example prompt generation based on metadata with robust error handling
        try:
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

            # グリッドサイズ
            grid_dimensions = metadata.get('grid_dimensions', None)
            if grid_dimensions is None or not isinstance(grid_dimensions, dict) or len(grid_dimensions) < 2:
                grid_dimensions = {
                    'width_grids': 10,
                    'height_grids': 10
                }
            
            # 縮尺
            scale_info = metadata.get('scale_info', None)
            if scale_info is None or not isinstance(scale_info, dict) or len(scale_info) < 2:
                scale_info = {
                    'drawing_scale': '1:100',
                    'grid_mm': 910,
                    'grid_px': 107.5
                }

            # 階数
            floor = metadata.get('floor', None)
            if floor is None:
                floor = '1F'

            # 建物のコンテキスト
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


            # 居室
            zones = metadata.get('zones', None)
            if zones is None or not isinstance(zones, list) or len(zones) < 2:
                zones = [
                    {'type': 'living', 'approximate_grids': 17, 'priority': 1},
                    {'type': 'private', 'approximate_grids': 14, 'priority': 2},
                    {'type': 'service', 'approximate_grids': 7, 'priority': 3},
                ]

            # プロンプト生成
            prompt_parts = []
            prompt_parts.append(f"grid_{annotation_metadata['grid_resolution']}")
            prompt_parts.append(f"area_{training_hints['total_area_grids']}grids")
            prompt_parts.append(f"scale_{scale_info['drawing_scale']}")
            prompt_parts.append(f"floor_{floor}")
            prompt_parts.append(f"rooms_{training_hints['room_count']}")

            if "entrance" in training_hints['floor_constraints']['prohibited_elements']:
                prompt_parts.append("entrance_prohibited")
            elif "entrance" in training_hints['floor_constraints']['required_elements'] or training_hints['has_entrance']:
                prompt_parts.append("entrance_required")

            if "stair" in training_hints['floor_constraints']['prohibited_elements']:
                prompt_parts.append("stair_prohibited")
            elif "stair" in training_hints['floor_constraints']['required_elements'] or training_hints['has_stair']:
                prompt_parts.append("stair_required")

            if "balcony" in training_hints['floor_constraints']['prohibited_elements']:
                prompt_parts.append("balcony_prohibited")
            elif "balcony" in training_hints['floor_constraints']['required_elements'] or training_hints['has_balcony']:
                prompt_parts.append("balcony_required")

            if 'living' in zones:
                idx_living = zones.index(next(z for z in zones if z['type'] == 'living'))
                prompt_parts.append(f"living_{zones[idx_living]['approximate_grids']}grids")
            if 'private' in zones:
                idx_private = zones.index(next(z for z in zones if z['type'] == 'private'))
                prompt_parts.append(f"private_{zones[idx_private]['approximate_grids']}grids")
            if 'service' in zones:
                idx_service = zones.index(next(z for z in zones if z['type'] == 'service'))
                prompt_parts.append(f"service_{zones[idx_service]['approximate_grids']}grids")

            prompt_parts.append("japanese_house")
            prompt_parts.append(f"{scale_info['grid_mm']}mm_grid")
            prompt_parts.append("architectural_plan")

            prompt = ", ".join(prompt_parts)
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