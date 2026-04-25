# HW03 人脸识别作业
基于 face_recognition(dlib) + Streamlit 实现

## 项目结构
HW03/
├── src/face_utils.py   # 人脸检测、特征提取、比对工具
├── known_faces/        # 自定义人脸库
├── app.py              # Streamlit 网页主程序
├── requirements.txt    # 依赖库
└── README.md           # 说明文档

## 功能
1. 人脸位置检测
2. 提取128维人脸特征编码
3. 已知人脸库比对识别
4. 网页端上传图片、可视化结果

## 安装依赖
pip install -r requirements.txt

## 运行命令
streamlit run app.py

## 访问
浏览器打开：http://localhost:8501

## 人脸库使用
将人名命名的图片放入 known_faces 文件夹即可自动识别
