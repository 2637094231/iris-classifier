# streamlit_iris/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========== 页面配置 ==========
st.set_page_config(
    page_title="Iris 花品种分类器",
    page_icon="🌺",
    layout="centered"
)

# ========== 标题和说明 ==========
st.title("🌺 Iris 花品种分类器")
st.markdown("""
使用机器学习模型预测鸢尾花的品种：
- **Setosa** (山鸢尾)
- **Versicolor** (变色鸢尾)  
- **Virginica** (维吉尼亚鸢尾)
""")

# ========== 加载模型 ==========
@st.cache_resource
def load_model(model_name):
    """加载指定的模型"""
    model_path = f"{model_name}_model_iris.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # 如果模型不存在，使用默认路径
        model_path = "logit_model_iris.pkl"
        return joblib.load(model_path)

# ========== 侧边栏：模型选择 ==========
st.sidebar.header("⚙️ 模型选择")

model_options = {
    "逻辑回归": "logit",
    "K近邻": "knn", 
    "支持向量机": "svm",
    "决策树": "dtree"
}

selected_model_name = st.sidebar.selectbox(
    "选择分类算法",
    list(model_options.keys())
)
model_key = model_options[selected_model_name]

# 加载模型
try:
    model = load_model(model_key)
    st.sidebar.success(f"✅ {selected_model_name} 模型已加载")
except Exception as e:
    st.sidebar.error(f"⚠️ 模型加载失败: {e}")
    model = None

# ========== 侧边栏：用户输入 ==========
st.sidebar.header("📊 输入花萼和花瓣参数")

def user_input_features():
    """获取用户输入的四个特征值"""
    sepal_length = st.sidebar.slider(
        "花萼长度 (Sepal Length) - cm",
        4.0, 8.0, 5.8, 0.1
    )
    
    sepal_width = st.sidebar.slider(
        "花萼宽度 (Sepal Width) - cm",
        2.0, 4.5, 3.0, 0.1
    )
    
    petal_length = st.sidebar.slider(
        "花瓣长度 (Petal Length) - cm",
        1.0, 7.0, 3.8, 0.1
    )
    
    petal_width = st.sidebar.slider(
        "花瓣宽度 (Petal Width) - cm",
        0.1, 2.5, 1.2, 0.1
    )
    
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ========== 主页面：显示输入值 ==========
st.subheader("📝 输入的参数值")

col1, col2 = st.columns(2)
with col1:
    st.metric("花萼长度", f"{input_df['sepal_length'].values[0]} cm")
    st.metric("花萼宽度", f"{input_df['sepal_width'].values[0]} cm")
with col2:
    st.metric("花瓣长度", f"{input_df['petal_length'].values[0]} cm")
    st.metric("花瓣宽度", f"{input_df['petal_width'].values[0]} cm")

# ========== 预测按钮和结果 ==========
st.markdown("---")

if st.button("🔮 预测鸢尾花品种", type="primary", use_container_width=True):
    if model is not None:
        # 进行预测
        features = input_df.values.reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # 调试信息（部署后可以删除）
        st.write(f"调试：预测原始值 = {prediction}, 类型 = {type(prediction)}")
        
        # 统一处理预测结果（兼容数字和字符串）
        # 将预测值转换为小写字符串
        pred_str = str(prediction).strip().lower()
        
        # 定义映射（同时支持数字和字符串）
        if pred_str in ["0", "setosa"]:
            result_name = "Setosa (山鸢尾)"
            color = "green"
        elif pred_str in ["1", "versicolor"]:
            result_name = "Versicolor (变色鸢尾)"
            color = "blue"
        elif pred_str in ["2", "virginica"]:
            result_name = "Virginica (维吉尼亚鸢尾)"
            color = "red"
        else:
            result_name = f"未知品种 (预测值: {prediction})"
            color = "gray"
        
        # 显示结果
        st.markdown("---")
        st.subheader("🎯 预测结果")
        
        # 使用彩色框显示结果
        st.markdown(
            f"""
            <div style="
                background-color: {color}20;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                border: 2px solid {color};
            ">
                <h2 style="color: {color}; margin: 0;">
                    {result_name}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 显示预测概率（如果模型支持且输出为数字）
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features)[0]
                st.subheader("📊 预测概率")
                
                prob_df = pd.DataFrame({
                    '品种': ['Setosa', 'Versicolor', 'Virginica'],
                    '概率': proba
                })
                st.bar_chart(prob_df.set_index('品种'))
            except:
                st.info("当前模型不支持概率预测")
    else:
        st.error("❌ 模型未加载，请检查模型文件是否存在")
# ========== 显示数据集预览 ==========
st.markdown("---")
with st.expander("📖 查看 Iris 数据集预览"):
    df = pd.read_csv("iris.csv")
    st.dataframe(df.head(10))
    st.caption(f"数据集共 {len(df)} 条记录")

# ========== 添加说明 ==========
st.markdown("---")
st.caption("💡 提示：在左侧边栏调整参数，点击预测按钮查看结果")
