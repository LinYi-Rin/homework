# HW07: 胸部X光肺炎二分类任务

本项目基于 Kaggle Chest X-Ray Images (Pneumonia) 数据集，实现肺炎二分类任务（Normal vs Pneumonia）。

## 运行环境与依赖
- Python 版本：`3.10+`
- 依赖包：见 `requirements.txt`

## 运行方式
1.  **Kaggle Notebook（推荐）**
    - 直接在 Kaggle 中新建 Notebook，通过 `Add Data` 挂载 `Chest X-Ray Images (Pneumonia)` 数据集
    - 上传 `train.ipynb` 并运行所有单元格

2.  **本地运行**
    ```bash
    pip install -r requirements.txt
    jupyter notebook hw07/train.ipynb
