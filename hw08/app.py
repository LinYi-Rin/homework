# 导入依赖库
import streamlit as st
import easyocr
from PIL import Image
import numpy as np

# ---------------------- 初始化EasyOCR视觉模型 ----------------------
# 开启中文+英文识别，CPU运行
reader = easyocr.Reader(['ch_sim','en'], gpu=False)

# ---------------------- 网页页面布局 ----------------------
st.set_page_config(page_title="图片文字识别OCR工具", page_icon="📝")
st.title("AI图文提取工具 | 计算机视觉OCR项目")
st.subheader("上传图片，一键提取图片内所有文字")

# 1. 图片上传模块
upload_file = st.file_uploader("请上传包含文字的图片", type=["jpg", "png", "jpeg"])

if upload_file is not None:
    # 读取上传图片
    img = Image.open(upload_file).convert("RGB")
    st.image(img, caption="上传原图", width=600)
    img_np = np.array(img)

    # 2. AI视觉推理核心（CV识别步骤）
    with st.spinner("AI正在识别图片文字，请稍候..."):
        result = reader.readtext(img_np)

    # 3. 解析识别结果
    full_text = ""
    for res in result:
        text, score = res[1], res[2]
        full_text += text + "\n"

    # 4. 展示识别结果
    st.divider()
    st.subheader("识别提取文字结果")
    st.text_area("完整文本", full_text, height=300)
    st.download_button("导出文字到txt文件", full_text, file_name="识别结果.txt")

else:
    st.info("请上传图片启动识别功能")
