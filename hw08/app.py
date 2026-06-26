# 导入依赖库
import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np

# ---------------------- 初始化AI视觉OCR模型（预训练CV模型） ----------------------
# use_angle_cls开启倾斜文字检测，lang=ch中文+英文识别
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)

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
        result = ocr.ocr(img_np, cls=True)

    # 3. 解析识别结果
    full_text = ""
    boxes = []
    txts = []
    scores = []
    for res in result[0]:
        box, text_info = res[0], res[1]
        text, score = text_info[0], text_info[1]
        full_text += text + "\n"
        boxes.append(box)
        txts.append(text)
        scores.append(score)

    # 4. 展示识别结果
    st.divider()
    st.subheader("识别提取文字结果")
    st.text_area("完整文本", full_text, height=300)
    st.download_button("导出文字到txt文件", full_text, file_name="识别结果.txt")

    # 绘制带文字框的效果图（可视化CV检测效果）
    font_path = "simhei.ttf"
    try:
        img_show = draw_ocr(img_np, boxes, txts, scores, font_path=font_path)
        img_show = Image.fromarray(img_show)
        st.image(img_show, caption="AI文字检测标注效果图")
    except:
        st.info("无字体文件，跳过标注图展示，不影响文字识别功能")

else:
    st.info("请上传图片启动识别功能")
