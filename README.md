# 910mmグリッド住宅プラン自動生成システム

MacBook Pro (M4 Max) とUbuntu 22.04上で動作する、建築図面PDFを学習し910mm/455mm混合グリッドで住宅平面図を自動生成するシステム。

## 📋 概要

- **目的**: 建築図面PDFから寸法を抽出し、AIで新しい平面図を生成
- **グリッド**: 910mm（本間）/ 455mm（半間）の日本建築標準寸法
- **出力**: FreeCADで編集可能な3Dモデル
- **処理時間目標**: 5秒以内/件

## 🏗️ システム構成

```
PDF図面 → 寸法抽出 → グリッド正規化 → AI学習 → 平面図生成 → 制約チェック → 3Dモデル
```

## 🚀 セットアップ

### 1. 環境準備（macOS）

```bash
cd ~/repos/floor_generate
brew install python@3.11 git cmake pkg-config poppler tesseract tesseract-lang
python3.11 -m venv floorplan_env
source floorplan_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
chmod +x setup_dirs.sh
./setup_dirs.sh
```

### 1-2. 環境準備（Ubuntu 22.04）

```bash
cd ~/repos/floor_generate
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y poppler-utils tesseract-ocr tesseract-ocr-jpn
sudo apt install -y cmake pkg-config git
python3.11 -m venv floorplan_env
source floorplan_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
chmod +x setup_dirs.sh
./setup_dirs.sh
```

### 2. 依存関係の維持・更新（Maintain Dependencies）

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --upgrade
pip list --outdated
```

## 📁 プロジェクト構造

```
floor_generate/
├── data/
│   ├── raw_pdfs/        # 元PDF図面
│   ├── extracted/       # 寸法抽出結果
│   ├── normalized/      # グリッド正規化済み
│   ├── training/        # 学習用データ
│   └── validation/      # 検証用データ
├── src/
│   ├── preprocessing/   # 前処理モジュール
│   ├── training/        # AI学習
│   ├── inference/       # 推論・生成
│   ├── constraints/     # 制約チェック
│   ├── freecad_bridge/  # FreeCAD連携
│   └── ui/             # Streamlit UI
├── models/              # 学習済みモデル
├── outputs/             # 生成結果
└── scripts/             # 実行スクリプト
```

## 🎯 使用方法（Setup Local App）

### 1. PDF図面の準備

85枚以上のPDF図面を `data/raw_pdfs/` ディレクトリに配置してください。

### 2. Streamlit UIの起動

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
streamlit run src/ui/main_app.py --server.port 8501 --server.address 0.0.0.0
```

### 3. PDFデータの前処理実行

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
python scripts/process_pdfs.py
```

### 4. 学習データ準備

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
python scripts/prepare_training_data.py --pdf_dir data/raw_pdfs --output_dir data/training
```

### 5. 完全なパイプラインテスト

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
python scripts/performance_test.py
```

### 6. 個別の平面図生成テスト

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
python scripts/generate_plan.py --width 11 --height 10 --output outputs/
```

### 7. 初回セットアップ確認

```bash
cd ~/repos/floor_generate
source floorplan_env/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import easyocr; print('EasyOCR imported successfully')"
python -c "import pdf2image; print('PDF2Image imported successfully')"
streamlit --version
```

### 結果の確認

抽出結果は `data/extracted/` に保存されます：
- 個別の寸法情報: `*_dimensions.json`
- 全体サマリー: `extraction_summary.json`

## 📊 現在の進捗

- [x] プロジェクト構造の作成
- [x] PDF寸法抽出モジュール
- [x] グリッド正規化システム
- [x] 学習データ生成骨格
- [x] AI学習システム骨格
- [x] 制約チェック骨格
- [x] FreeCAD連携骨格
- [x] UI実装骨格
- [ ] 学習データ生成詳細実装
- [ ] パイプライン統合テスト

## 🔧 トラブルシューティング

### OCRが動作しない場合（macOS）

```bash
tesseract --version
brew install tesseract-lang
```

### OCRが動作しない場合（Ubuntu）

```bash
tesseract --version
sudo apt install tesseract-ocr-jpn
```

### PDFが読み込めない場合

```bash
pdftoppm -h
python -c "import pdf2image; print('PDF processing available')"
```

### 仮想環境の問題

```bash
cd ~/repos/floor_generate
rm -rf floorplan_env
python3.11 -m venv floorplan_env
source floorplan_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 📝 ライセンス

このプロジェクトはMITライセンスで公開されています。

## 🤝 貢献

プルリクエストを歓迎します！

## 📞 連絡先

質問や提案がある場合は、Issueを作成してください。
