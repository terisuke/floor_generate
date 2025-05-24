## **MVP要件定義書: 910mmグリッド住宅プラン自動生成システム**

### **1. プロジェクト概要**

#### **🎯 目的**
建築図面PDFを学習データとし、日本の建築基準で一般的な910mm/455mm混合グリッドシステムに基づいて、AIが住宅の平面図を自動生成する。生成されたプランは基本的な建築的整合性を持ち、FreeCADで編集可能な2D/3Dデータとして出力することを目指す [cite: 1, 2]。

#### **✅ MVPにおける主要成功基準**
1.  **処理時間**: ユーザー入力から2D平面図および基本的な3Dモデルの生成・表示まで、**2秒以内/件**を目指す（M4 Max環境を想定） [cite: 1, 2]。
2.  **データ処理パイプラインの確立**:
    * `data/raw_pdfs/` 内のPDFから、PaddleOCR等を用いてテキスト情報（LDKタイプ、面積等）および主要寸法を抽出する機能を実装する [cite: 1, 2]。
    * 抽出された寸法を910mm/455mmグリッドに正規化する機能を実装する [cite: 2]。
    * **MVPの焦点**: 上記からAI学習用の「入力（プロンプト、敷地マスク）」と「出力（ターゲット平面図画像）」のペアを**半自動的に**生成する基盤を構築する。初期は壁情報を主とし、限定的なPDFセットで動作検証を行う。目標は3k-5kペアだが、MVPではこのパイプライン構築と少量データでの動作実証を優先する [cite: 1, 2]。
3.  **AIによる平面図生成**:
    * Stable DiffusionベースのモデルとLoRAファインチューニングを用い、指定された敷地条件（マスク画像）とテキストプロンプトに基づき、壁を中心とした基本的な住宅平面図（例: 256x256pxの多チャンネル画像）を生成できる [cite: 2]。
    * 生成品質: 視覚的に認識可能な壁構造と、基本的な部屋の区画が生成されること。壁閉合率等の厳密な数値目標はMVP後の改善とする。
4.  **基本的な制約チェック**:
    * 生成されたプランに対し、浮遊壁がないかなど、ごく基本的な構造的整合性チェックを行う（CP-SATの枠組みは用意しつつ、MVPではルールベースの簡易チェックを優先） [cite: 2]。
5.  **CAD連携 (2D/3D出力)**:
    * AIが生成したプラン（壁情報）から、FreeCADで壁厚を持ったシンプルな3Dモデルを自動生成し、`.FCStd`形式で保存できる [cite: 2]。
    * 生成された2DプランをSVG形式でエクスポートできる [cite: 2]。
6.  **ユーザーインターフェース (Streamlit)**:
    * ユーザーが敷地サイズ（グリッド単位）や基本的なプラン要件（LDKタイプ、スタイル等）を入力できる [cite: 2]。
    * 生成された2D平面図プレビューと、3Dモデル（`.FCStd`）のダウンロードリンクを提供する [cite: 2]。

#### **🏗️ MVP対象建物**
* 2階建て在来木造住宅（日本標準仕様） [cite: 2]。
* 敷地: 矩形（例: 8×6〜15×12グリッド） [cite: 2]。
* 部屋数: 3LDK〜5LDKの範囲を想定した入力に対応 [cite: 2]。
* 延床面積: 80〜140㎡の範囲を想定 [cite: 2]。
    (MVPでは、生成されるプランがこれらの条件を完全に満たすことよりも、条件を入力として受け付け、それらしい構造を生成するパイプラインの確立を優先)

---

### **2. システム構成**

```mermaid
graph TD
    A[PDF図面集<br>目標3k-5kペア<br>(MVPでは少量で実証)] --> B[寸法・特徴抽出<br>src/preprocessing/<br>dimension_extractor.py<br>(PaddleOCR)]
    B --> C[グリッド正規化<br>src/preprocessing/<br>grid_normalizer.py<br>(910mm + 455mm)]
    C --> D[学習データペア生成<br>src/preprocessing/<br>training_data_generator.py<br>(半自動化: 壁情報中心)]
    D --> E[学習データセット<br>data/training/]
    E --> F[AIモデル学習<br>src/training/<br>lora_trainer.py<br>(SD + LoRA)]
    
    G[Streamlit UI<br>src/ui/main_app.py<br>(敷地グリッド・LDK等入力)] --> H[敷地マスク生成<br>src/inference/generator.py]
    H --> I[AI推論<br>src/inference/generator.py<br>(平面図生成)]
    I --> J[基本制約チェック<br>src/constraints/<br>architectural_constraints.py<br>(ルールベース優先)]
    J --> K[ベクタ変換<br>(SVG出力)]
    K --> L[FreeCAD連携<br>src/freecad_bridge/<br>fcstd_generator.py<br>(壁の3D押出)]
    L --> N[3Dモデル<br>outputs/freecad/*.FCStd]
```
[cite: 1, 2]

### **3. MVPにおけるコア機能とモジュール**

* **データ前処理パイプライン (`src/preprocessing/`)**:
    * `dimension_extractor.py`: PDFからPaddleOCRを使いテキスト情報（LDK、面積、特徴等）と数値を抽出。プロンプトと敷地マスク生成の元情報とする [cite: 1, 2]。
    * `grid_normalizer.py`: 抽出寸法を910mm/455mm混合グリッドに正規化 [cite: 2]。
    * `training_data_generator.py`: **(MVP最重要課題)** PDFから抽出・正規化された情報を元に、壁情報を中心としたターゲット平面図画像（例: 256x256, チャンネル別）と、対応するプロンプト・敷地マスクを生成する**半自動化**パイプラインの基盤を構築する。初期は手動介入を許容しつつ、徐々に自動化範囲を拡大する。
* **AIモデル学習 (`src/training/`)**:
    * `lora_trainer.py`: Stable Diffusion (v1.4またはv2.1) をベースにLoRAファインチューニングを実行 [cite: 2]。
    * `dataset.py`: `data/training/` 内の学習データペアを読み込むデータローダー [cite: 2]。
* **AI推論 (`src/inference/generator.py`)**:
    * 学習済みLoRAモデルを使用し、UIからの入力（敷地マスク、プロンプト）に基づいて平面図画像を生成する [cite: 2]。
* **制約チェック (`src/constraints/architectural_constraints.py`)**:
    * MVPでは、生成されたプランの壁の連続性など、ごく基本的なルールベースのチェックを実装。CP-SATの枠組みは用意するが、複雑な解決はMVP後とする [cite: 2]。
* **FreeCAD連携 (`src/freecad_bridge/fcstd_generator.py`)**:
    * AI生成プランの壁情報に基づき、指定された壁厚で3D形状を押し出し、`.FCStd`ファイルとして保存する。SVG形式での2Dエクスポートも行う [cite: 2]。
* **UI (`src/ui/main_app.py`)**:
    * Streamlitを使用し、ユーザー入力、プラン生成実行、結果（2Dプレビュー、3Dファイルダウンロード）表示を行う [cite: 2]。

### **4. 主要技術スタックと開発環境**
* **プログラミング言語**: Python 3.11 [cite: 1, 2]。
* **AIフレームワーク**: PyTorch (2.3.1.dev* 等、M4 Max MPS対応の最新ナイトリー版推奨) [cite: 2]。
* **画像生成モデル**: Diffusersライブラリ経由のStable Diffusion + LoRA (PEFTライブラリ) [cite: 2]。
* **OCR**: PaddleOCR (日本語対応)、PaddlePaddle [cite: 1]。
* **画像処理**: OpenCV, Pillow [cite: 2]。
* **UI**: Streamlit [cite: 2]。
* **CAD連携**: FreeCAD 0.22 (Python API利用、公式AppImage/バンドル推奨) [cite: 2]。
* **対象ハードウェア**: MacBook Pro (M4 Max), Ubuntu 22.04 [cite: 1, 2]。

### **5. MVPにおける非目標（スコープ外）**
* 多様なPDF形式からの全建築要素（窓、ドア、設備記号、複雑な部屋形状等）の完全自動かつ高精度な抽出・正規化。MVPでは壁情報を優先し、他は簡略化または手動補完を許容する。
* 高品質で詳細なフォトリアリスティック3Dモデルの自動生成。MVPでは基本的な壁の押し出しに留める。
* FreeCAD上での高度なパラメトリック編集機能の完全実装。
* 複雑な建築法規や構造計算、高度な動線計画を含む制約チェックと自動修正。
* 複数階層プランの同時生成・3D統合（MVPでは1フロアごとの処理を基本とし、2階建ての概念は主にプロンプトと面積情報で扱う）。
* 初期MVP納品時点での3000〜5000ペアの大規模データセットによる完全学習済みモデルの提供（パイプラインの設計と少量データでの実証を優先）。

### **6. 開発・テスト**
* 既存のスクリプト群 (`scripts/`) を活用し、データ準備 (`prepare_training_data.py`)、モデル学習 (`train_model.py`)、推論 (`generate_plan.py`)、E2Eテスト (`train_and_display.py`, `performance_test.py`をMVPスコープに合わせて調整) のパイプラインを整備する [cite: 1, 2]。
* 主要な前処理機能（寸法抽出、グリッド正規化）、AI生成品質（壁構造）、CAD出力に関して基本的なテストを行う。

---

このMVP要件定義は、現行の`README.md`と`comprehensive_mvp_requirements.md` v1.1の情報を統合し、特に**データ前処理の自動化（半自動化からの段階的改善）**と**コアとなるAI生成パイプラインの確立**に焦点を当てています。最も困難な教師データ生成部分については、初期のMVPでは限定的な自動化と手動サポートを許容しつつ、将来的な拡張性を持たせた基盤を作ることを目指します。
