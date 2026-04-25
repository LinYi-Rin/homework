import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from src.face_utils import load_known_faces, detect_faces, get_face_encodings, recognize_faces

st.set_page_config(page_title="人脸识别作业", layout="wide")
st.title("👤 基于face_recognition的人脸识别系统")

@st.cache_resource
def load_lib():
    return load_known_faces()

known_encodings, known_names = load_lib()
st.sidebar.info(f"已加载人脸库数量：{len(known_names)}")

st.subheader("上传图片")
uploaded_file = st.file_uploader("选择人脸图片", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="原始图片", use_column_width=True)

    # 1.人脸检测
    face_locations = detect_faces(image)
    # 2.提取128维特征
    face_encodings = get_face_encodings(image, face_locations)
    # 3.人脸识别比对
    face_names = recognize_faces(face_encodings, known_encodings, known_names)

    # 画人脸框
    img_box = image.copy()
    draw = ImageDraw.Draw(img_box)
    for (top,right,bottom,left), name in zip(face_locations, face_names):
        color = "green" if name != "Unknown" else "red"
        draw.rectangle([left,top,right,bottom], outline=color, width=3)
        draw.text((left, top-10), name, fill=color)

    with col2:
        st.image(img_box, caption="检测+识别结果", use_column_width=True)

    st.success(f"一共检测到 {len(face_locations)} 张人脸")

    # 展示128维特征
    st.subheader("人脸128维特征编码")
    for idx, feat in enumerate(face_encodings):
        with st.expander(f"第{idx+1}个人脸 特征向量"):
            st.write(feat.tolist())
