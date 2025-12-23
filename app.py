import streamlit as st
import torch
import os
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import matplotlib.pyplot as plt

# --- 1. 配置與路徑設定 ---
# 動態偵測模型路徑，確保在不同電腦上都能執行
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'best_ai_detector_model')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data.csv')

# 檢查設備 (GPU 或 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 核心功能：模型載入 (使用快取) ---
@st.cache_resource
def load_ai_model(path):
    """載入訓練好的 Transformers 模型與 Tokenizer"""
    if not os.path.exists(path):
        st.error(f"❌ 找不到模型資料夾於：{path}\n請確認已執行訓練腳本並產生模型。")
        st.stop()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(DEVICE)
        model.eval()  # 設定為評估模式
        return tokenizer, model
    except Exception as e:
        st.error(f"模型載入失敗：{e}")
        st.stop()

@st.cache_data
def load_test_dataset(path):
    """載入用於 Demo 驗證的測試集 CSV"""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# 初始化模型與資料
tokenizer, model = load_ai_model(MODEL_PATH)
test_df = load_test_dataset(TEST_DATA_PATH)

# --- 3. Streamlit 介面設計 ---
st.set_page_config(page_title="🤖 AI 文章偵測器", layout="centered")

st.title("🤖 AI / Human 文章偵測器")
st.markdown("""
本工具利用 **RoBERTa** 深度學習模型，分析文本的語義特徵，判斷其是由人工撰寫還是 AI 生成。
""")

# --- 4. Demo 隨機抽選功能 ---
st.subheader("📝 步驟一：輸入或抽選文本")

# 初始化 session_state 以存儲抽選內容
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'actual_label' not in st.session_state:
    st.session_state.actual_label = None

if test_df is not None:
    if st.button("🎲 隨機從測試集 (Test Set) 抽選一篇文章"):
        # 隨機選取一列
        sample = test_df.sample(1).iloc[0]
        st.session_state.input_text = sample['text']
        st.session_state.actual_label = "AI 生成" if sample['label'] == 1 else "人類撰寫"
        st.rerun() # 重新整理以更新文字框
else:
    st.info("提示：若要使用隨機抽選功能，請先執行 split_data.py 產生 test_data.csv。")

# 文本輸入區
user_input = st.text_area(
    "請輸入待分析的內容：", 
    value=st.session_state.input_text, 
    height=250, 
    placeholder="在此輸入文章段落..."
)

# 若有抽選內容，顯示真實標籤以供對比
if st.session_state.actual_label:
    st.info(f"📍 **資料庫真實標籤：{st.session_state.actual_label}** (僅供 Demo 驗證對比)")

# --- 5. 偵測與結果顯示 ---
if st.button("🔍 執行 AI 偵測", type="primary"):
    if user_input.strip():
        with st.spinner("模型正在深度分析中，請稍候..."):
            # 1. 預處理文本
            inputs = tokenizer(
                user_input, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(DEVICE)
            
            # 2. 模型推理
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用 Softmax 轉化為概率
                probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
            human_prob = probs[0]
            ai_prob = probs[1]

            # 3. 統計量與可視化展示
            st.divider()
            st.subheader("📊 文本特徵統計與預測")
            
            # 第一層：核心指標
            col1, col2, col3 = st.columns(3)
            col1.metric("🧑🏻 人類概率", f"{human_prob:.2%}")
            col2.metric("🤖 AI 概率", f"{ai_prob:.2%}")
            col3.metric("📝 字數統計", len(user_input.split()))

            # 第二層：概率分佈條
            st.write("**模型信心分佈圖：**")
            chart_data = pd.DataFrame({
                "來源": ["人類 (Human)", "人工智慧 (AI)"],
                "機率 (%)": [human_prob * 100, ai_prob * 100]
            })
            st.bar_chart(chart_data.set_index("來源"))

            # 第三層：結論報告
            if ai_prob > 0.5:
                st.warning(f"🚨 **判定結論：高度疑似為 AI 生成內容**")
                st.info(f"模型分析顯示，該文本具有明顯的語言模型特徵，AI 信心度為 {ai_prob:.1%}")
            else:
                st.success(f"✅ **判定結論：高度疑似為人類撰寫內容**")
                st.info(f"模型分析顯示，該文本語義流動較符合人類習慣，人類信心度為 {human_prob:.1%}")

            # 若有抽選資料，顯示標籤對比 (增加統計可信度)
            if st.session_state.actual_label:
                st.markdown(f"**驗證對比：** 真實標籤為 `{st.session_state.actual_label}`")
    else:
        st.error("請輸入內容後再執行偵測！")

# --- 6. 進階分析：SHAP 解釋器 ---
st.subheader("🔍 進階特徵分析 (可解釋 AI)")
if st.button("🧬 執行 SHAP 關鍵字分析"):
    if user_input.strip():
        with st.spinner("正在計算單詞貢獻度，這可能需要幾十秒..."):
            try:
                # 修正後的預測函數
                def predict_probs(texts):
                    # SHAP 有時會傳入 numpy 陣列，需確保轉為 list
                    texts = [str(t) for t in texts] 
                    inputs = tokenizer(
                        texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    ).to(DEVICE)
                    
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        # SHAP 需要的是機率值 (Probability)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    return probs

                # 使用 shap.maskers.Text 處理 Tokenizer 確保對齊
                masker = shap.maskers.Text(tokenizer)
                
                # 初始化解釋器
                explainer = shap.Explainer(predict_probs, masker=masker, output_names=["Human", "AI"])
                
                # 計算 SHAP 值
                shap_values = explainer([user_input])

                # 視覺化輸出
                st.write("**關鍵詞影響力分析：**")
                st.caption("🔴 紅色：增加 AI 生成疑慮 | 🔵 藍色：偏向人類撰寫特徵")
                
                # 使用 HTML 渲染 SHAP 結果
                shap_html = shap.plots.text(shap_values[0], display=False)
                st.components.v1.html(shap_html, height=400, scrolling=True)
                
            except Exception as e:
                st.error(f"SHAP 分析發生錯誤：{e}")
                st.info("提示：這可能是因為文本過短或包含特殊字元，請嘗試更換一段文字。")
    else:
        st.error("請先輸入文字再執行分析！")

# 頁尾說明
st.divider()
st.caption("技術底層：Transformers RoBERTa-base | 數據集：train_v2_drcat_02")