# floor_generateプロジェクトでの統合JSON利用ガイド

## 概要

`pdf-vector-crop-rasterizer`で生成される統合JSONファイルを`floor_generate`プロジェクトで利用するための設定ガイドです。

## 1. 統合JSONファイルの配置

### 推奨ディレクトリ構造
```
floor_generate/
├── data/
│   ├── training/
│   │   ├── integrated/      # 統合JSONファイルを配置
│   │   │   ├── plan_xxx_1f_integrated.json
│   │   │   ├── plan_xxx_2f_integrated.json
│   │   │   └── ...
│   │   ├── images/          # 対応する画像ファイル
│   │   │   ├── plan_xxx_1f.png
│   │   │   ├── plan_xxx_2f.png
│   │   │   └── ...
```

## 2. データセットクラスの設定

### 2.1 基本的な使用方法

```python
from src.training.dataset import FloorPlanDataset

# 統合JSONを優先的に使用（デフォルト）
dataset = FloorPlanDataset(
    data_dir="data/training",
    use_integrated_json=True  # デフォルトでTrue
)

# 統計情報の確認
stats = dataset.get_data_statistics()
print(f"Total pairs: {stats['total_pairs']}")
print(f"Data types: {stats['data_types']}")
print(f"Grid sizes: {stats['grid_sizes']}")
```

### 2.2 従来形式との互換性

既存のPhase 1/2形式やディレクトリ形式のデータも自動的に読み込まれます：

```python
# 混在環境での使用
# - integrated/ ディレクトリの統合JSON
# - 従来のPhase 1/2ペア
# - pair_xxxx/ ディレクトリ形式
# すべて自動的に読み込まれます
dataset = FloorPlanDataset("data/training")
```

## 3. 学習スクリプトの更新

### 3.1 train_model.py の例

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import patch_diffusers
patch_diffusers.apply_patches()

from src.training.dataset import FloorPlanDataset
from src.training.lora_trainer import LoRATrainer
from torch.utils.data import DataLoader

def train():
    # データセットの準備
    train_dataset = FloorPlanDataset(
        data_dir="data/training",
        use_integrated_json=True
    )
    
    # データセットの統計を表示
    stats = train_dataset.get_data_statistics()
    print("Dataset Statistics:")
    print(f"  Total samples: {stats['total_pairs']}")
    print(f"  Grid sizes: {stats['grid_sizes']}")
    print(f"  Floor distribution: {stats['floors']}")
    
    # DataLoaderの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # トレーナーの初期化と学習
    trainer = LoRATrainer()
    trainer.train(train_loader, num_epochs=20)

if __name__ == "__main__":
    train()
```

## 4. プロンプト生成の改善

統合JSONにより、より詳細なプロンプトが自動生成されます：

### 4.1 生成されるプロンプトの例

```
# 統合JSONからの自動生成プロンプト
"grid_6x10, area_60grids, scale_1:100, module_910mm, floor_1F, entrance_1, stair_1, rooms_4, entrance_required, ground_floor, living_17grids, private_14grids, japanese_house, architectural_plan, 910mm_grid"
```

### 4.2 カスタムプロンプト生成

必要に応じて`dataset.py`の`generate_enhanced_prompt`メソッドをカスタマイズできます：

```python
def generate_enhanced_prompt(self, metadata: Dict) -> str:
    """カスタマイズ例"""
    prompt_parts = []
    
    # 基本情報
    grid_dims = metadata.get('grid_dimensions', {})
    prompt_parts.append(f"size_{grid_dims.get('width_grids')}x{grid_dims.get('height_grids')}")
    
    # 建築様式の追加
    style = metadata.get('building_context', {}).get('style', 'modern')
    prompt_parts.append(f"style_{style}")
    
    # その他のカスタム条件
    # ...
    
    return ', '.join(prompt_parts)
```

## 5. 推論時の使用方法

### 5.1 FloorPlanGeneratorでの活用

```python
class FloorPlanGenerator:
    def __init__(self, model_path: str):
        self.trainer = LoRATrainer()
        self.trainer.load_lora_weights(model_path)
        self.pipeline = self.trainer.get_pipeline_for_inference()
    
    def generate_plan(self, site_mask, conditions: Dict):
        """
        conditions: {
            'grid_dimensions': {'width_grids': 6, 'height_grids': 10},
            'floor': '1F',
            'scale_info': {'grid_mm': 910, 'drawing_scale': '1:100'},
            'room_count': 4
        }
        """
        # 条件からプロンプトを生成
        prompt = self.build_prompt_from_conditions(conditions)
        
        # 生成実行
        generated = self.pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        return generated
    
    def build_prompt_from_conditions(self, conditions: Dict) -> str:
        """条件からプロンプトを構築"""
        parts = []
        
        grid = conditions.get('grid_dimensions', {})
        if grid:
            parts.append(f"grid_{grid['width_grids']}x{grid['height_grids']}")
        
        floor = conditions.get('floor', '1F')
        parts.append(f"floor_{floor}")
        
        # ... その他の条件
        
        parts.extend(['japanese_house', 'architectural_plan', '910mm_grid'])
        return ', '.join(parts)
```

## 6. トラブルシューティング

### 6.1 統合JSONが読み込まれない場合

```python
# デバッグモードで確認
dataset = FloorPlanDataset("data/training")
stats = dataset.get_data_statistics()

# データタイプ別の内訳を確認
print("Data types found:")
for dtype, count in stats['data_types'].items():
    print(f"  {dtype}: {count}")
```

### 6.2 画像ファイルが見つからない場合

統合JSONファイルと画像ファイルの命名規則を確認：
- 統合JSON: `plan_xxx_1f_integrated.json`
- 画像ファイル: `plan_xxx_1f.png`

ベース名（`plan_xxx_1f`）が一致している必要があります。

## 7. ベストプラクティス

1. **データの整理**: 統合JSONファイルは専用のディレクトリ（`integrated/`）に配置
2. **命名規則の統一**: ファイル名の一貫性を保つ
3. **メタデータの検証**: 学習前にデータセットの統計情報を確認
4. **段階的な移行**: 既存データと新規データを混在させながら段階的に移行

## 8. 性能向上のヒント

### 8.1 条件付け強度の調整

```python
# より強い条件付けのためのプロンプト重み付け
prompt = "grid_6x10:1.2, floor_1F:1.5, entrance_required:2.0, ..."
```

### 8.2 データ拡張

```python
# 同じ間取りの異なる表現を学習
augmented_prompts = [
    "6x10_grid, 1F, entrance_south, ...",
    "grid_6x10, first_floor, main_entrance, ...",
    "60_grids_total, ground_floor, entry_1, ..."
]
```

## まとめ

統合JSONファイルにより、Diffusionモデルの学習において以下の改善が期待できます：

1. **より正確な条件付け**: グリッドサイズ、階数、スケールなどの詳細な制御
2. **一貫性のある生成**: メタデータに基づく制約の遵守
3. **効率的な学習**: 構造化されたデータによる学習効率の向上

詳細な実装については、`src/training/dataset.py`のコメントを参照してください。
