# 統合JSONを活用した高精度プロンプト生成ガイド

## 概要

`plan_118_1f_integrated.json`のような統合JSONデータを活用することで、Stable Diffusionの条件付け精度を大幅に向上させることができます。

## プロンプト生成戦略

### 1. 基本構造化プロンプト
```
grid_6x10, stair_0.2_3.5_entrance_3.7_8.3, total_area_60grids, room_count_3, floor_2F, scale_1to100, japanese_residential, 910mm_grid
```

**特徴**:
- グリッド寸法の正確な指定
- 構造要素の座標情報
- 面積・部屋数の定量化

### 2. 拡張コンテキストプロンプト
```
building_single_family_house_2floors, current_floor_2F, grid_dimensions_6x10, structural_elements_stair_grid0.2x3.5_entrance_grid3.7x8.3, zones_living17grids_private14grids_service7grids, stair_alignment_critical, drawing_scale_1to100, japanese_residential_910mm_grid, architectural_plan
```

**特徴**:
- 建築コンテキストの詳細化
- ゾーン配置の優先度情報
- 制約条件の明示

### 3. 階層的条件付けプロンプト
```
floor_2F: stair+private_sleeping_areas+work_space+utility_area+balcony, grid_6x10_60total, stair_u_turn_1.6x0.7grids_pos0.2x3.5, entrance_1.3x1.1grids_pos3.7x8.3, japanese_house_910mm_standard
```

**特徴**:
- フロア機能の階層的表現
- 要素サイズの詳細指定
- 建築標準の明示

## 実装済み機能

`FloorPlanDataset.generate_prompt()`メソッドが以下の機能で拡張されました：

### 統合JSON対応
- `grid_dimensions`からの正確なグリッド情報
- `structural_elements`の座標データ活用
- `building_context`の建築情報統合
- `zones`の配置優先度反映

### 後方互換性
- 従来のメタデータフォーマットも継続サポート
- 自動フォーマット検出機能

## 精度向上のメカニズム

### 1. 空間的精度
- グリッド座標による正確な要素配置
- 910mm標準モジュールの厳密な適用

### 2. 建築的制約
- フロア機能の階層的分離
- 構造要素の配置ルール

### 3. 条件付け強度
- 詳細メタデータによる強い条件付け
- CFG (Classifier-Free Guidance) の効果的活用

## 使用方法

### 学習データセットでの使用
```python
# 統合JSONデータでの学習
dataset = FloorPlanDataset(data_dir="data/training")
sample = dataset[0]
print(sample['prompt'])
# 出力: "building_single_family_house_2floors, current_floor_2F, grid_6x10, ..."
```

### 生成スクリプトでの使用
```bash
# 従来のプロンプト生成
python scripts/generate_plan.py --width 6 --height 10 --rooms 3LDK

# 拡張プロンプト生成
python scripts/generate_plan.py --width 6 --height 10 --rooms 3LDK --enhanced_prompt --current_floor 1F --entrance_x 3.7 --entrance_y 8.3

# 2階の生成例
python scripts/generate_plan.py --width 6 --height 10 --rooms 3LDK --enhanced_prompt --current_floor 2F --floors 2
```

## 期待される効果

1. **生成精度の向上**: 構造要素の正確な配置
2. **建築制約の遵守**: 日本建築基準への準拠
3. **学習効率の改善**: 詳細な条件付けによる収束速度向上
4. **品質の一貫性**: 標準化されたプロンプト形式

この拡張により、従来の基本的なプロンプトから、建築的に意味のある詳細な条件付けが可能になります。
