# 910mmグリッド住宅プラン自動生成システム

MacBook Pro (M4 Max) 上で動作する、建築図面PDFを学習し910mm/455mm混合グリッドで住宅平面図を自動生成するシステム。

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

### 1. 環境準備

```bash
# Python 3.11 仮想環境の作成
python3.11 -m venv venv
source venv/bin/activate  # Mac/Linux

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 追加ツールのインストール

```bash
# Homebrew でツールをインストール
brew install poppler tesseract
```

### 3. ディレクトリ構造の作成

```bash
# セットアップスクリプトの実行
chmod +x setup_dirs.sh
./setup_dirs.sh
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

## 🎯 使用方法

### 1. PDF図面の準備

6枚のPDF図面を `data/raw_pdfs/` ディレクトリに配置してください。

### 2. 寸法抽出の実行

```bash
# PDFから寸法を抽出
python scripts/process_pdfs.py
```

### 3. 結果の確認

抽出結果は `data/extracted/` に保存されます：
- 個別の寸法情報: `*_dimensions.json`
- 全体サマリー: `extraction_summary.json`

## 📊 現在の進捗

- [x] プロジェクト構造の作成
- [x] PDF寸法抽出モジュール
- [x] グリッド正規化システム
- [ ] 学習データ生成
- [ ] AI学習システム
- [ ] 制約チェック
- [ ] FreeCAD連携
- [ ] UI実装

## 🔧 トラブルシューティング

### OCRが動作しない場合

```bash
# Tesseractの確認
tesseract --version

# 日本語データのインストール
brew install tesseract-lang
```

### PDFが読み込めない場合

```bash
# Popplerの確認
pdftoppm -h
```

## 📝 ライセンス

このプロジェクトはMITライセンスで公開されています。

## 🤝 貢献

プルリクエストを歓迎します！

## 📞 連絡先

質問や提案がある場合は、Issueを作成してください。
