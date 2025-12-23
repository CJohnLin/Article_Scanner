# 🤖 AI vs. Human 文本偵測器

這是一個基於深度學習的自然語言處理（NLP）專案，旨在準確辨識一段文本是由 **人類撰寫** 還是由 **AI 生成**（如 ChatGPT, Claude 等）。本專案使用 **RoBERTa-base** 模型進行微調，並提供一個直觀的 Streamlit 網頁操作介面。

## 🌟 核心功能
* **高精度偵測**：利用在 `train_v2_drcat_02` 數據集上微調的 RoBERTa 模型。
* **隨機驗證系統**：可從獨立的測試集 (`test_data.csv`) 中一鍵隨機抽取文章，即時驗證模型準確度。
* **概率可視化**：以直觀的計量表與圖表呈現「人類」與「AI」的信心得分。
* **即時推理**：支援手動輸入自定義文本進行即時分析。

## 🛠️ 技術架構
* **模型底層**：Transformers `roberta-base`
* **開發框架**：PyTorch, Hugging Face `Trainer`
* **數據處理**：Pandas, Scikit-learn, Git LFS
* **網頁介面**：Streamlit

## 🚀 快速開始

### 1. 複製倉庫
\`\`\`bash
git clone https://github.com/CJohnLin/AI_VS_HUMAN.git
cd AI_VS_HUMAN
\`\`\`

### 2. 安裝依賴項
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. 下載模型權重
由於模型檔案較大（約 486MB），請確保已安裝 **Git LFS** 以正確拉取 \`best_ai_detector_model/\` 中的 \`model.safetensors\`。
\`\`\`bash
git lfs install
git lfs pull
\`\`\`

### 4. 啟動應用程式
\`\`\`bash
streamlit run app.py
\`\`\`

### 🧬 模型可解釋性：SHAP 關鍵字分析 (Model Interpretability)
為了提升偵測器的透明度，本專案導入了 SHAP (SHapley Additive exPlanations) 技術。當使用者執行「進階特徵分析」時，系統會計算每個單詞對預測結果的貢獻度：

#### 紅色標記 (Red)：代表該詞彙增加了模型判定為 AI 生成 的疑慮。

#### 藍色標記 (Blue)：代表該詞彙具有明顯的 人類撰寫 特徵。

基礎值 (Base Value)：圖表上方的數值代表模型在看到任何文字前的初始信心度（本模型對 AI 文章的基礎判定信心極高，約為 0.99）。

#### 📈 統計量說明
本 App 提供以下關鍵統計指標：

AI/Human 概率百分比：由 RoBERTa 模型最後一層 Softmax 產出的信心度指標。

字數統計 (Word Count)：分析文本長度對偵測結果穩定性的影響。

SHAP 貢獻分佈圖：視覺化展示模型判斷的核心依據詞彙。

## ⚠️ 運行限制與須知 (Important)

在使用或部署本專案時，請注意以下限制：

### 1. 硬體資源需求 (Local Running)
* **記憶體 (RAM)**：模型載入與推理建議至少具備 **8GB RAM**。文書筆電若開啟過多分頁可能導致 \`app.py\` 崩潰。
* **運算資源**：本應用預設會自動偵測並使用 GPU（CUDA），若無 GPU 則使用 CPU 運算，單次推理時間約 2-5 秒。

### 2. 長文本截斷
* 模型輸入上限為 **512 tokens**。若輸入文章過長，系統會自動截取前半段進行判斷，可能影響長篇論文後半段的偵測準確度。

### 3. 線上部署限制 (Streamlit Cloud)
* **記憶體溢出 (OOM)**：Streamlit Cloud 的免費層級記憶體有限，載入大型模型權重時可能觸發重啟。
* **Git LFS 配額**：GitHub LFS 的免費流量為每月 1GB。若頻繁 Clone 或線上載入，可能會導致數據集無法下載。

## 📁 檔案結構說明
* \`app.py\`: Streamlit 網頁應用程式主程式。
* \`best_ai_detector_model/\`: 訓練完成的模型權重與設定檔。
* \`fine_tuning_script.py\`: 用於訓練與微調模型的腳本。
* \`test_data.csv\`: 從原始數據分割出的獨立測試集，用於 Demo 驗證。
* \`requirements.txt\`: 專案執行所需的 Python 函式庫列表。

## 📧 聯絡資訊
作者：CJohnLin  
GitHub: [https://github.com/CJohnLin/AI_VS_HUMAN](https://github.com/CJohnLin/AI_VS_HUMAN)
EOF
