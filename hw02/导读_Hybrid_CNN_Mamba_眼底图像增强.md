# 论文导读：Hybrid CNN-Mamba for Multi-Scale Fundus Image Enhancement
- 论文出处：XXX Conference 2024
- 使用大模型：DeepSeek V3

## 1. 研究背景与动机
- 眼底图像是眼科疾病诊断的重要依据，但存在分辨率低、对比度差等问题
- 传统CNN难以捕捉长程依赖，Mamba模型在序列建模上效率更高
- 本文提出Hybrid CNN-Mamba架构，结合两者优势提升眼底图像增强效果

## 2. 核心方法
![Hybrid CNN-Mamba 模型架构图](figures/architecture.png)
*图1 本文提出的Hybrid CNN-Mamba模型架构，包含CNN局部特征提取分支与Mamba全局建模分支*
- **CNN分支**：提取局部纹理、边缘等细粒度特征
- **Mamba分支**：建模全局上下文与长程依赖关系
- **特征融合模块**：加权融合两个分支的输出
- 损失函数：感知损失 + SSIM + MSE

## 3. 主要结果
- 在公开眼底数据集上，PSNR/SSIM指标优于现有SOTA
- 可视化对比：增强后图像细节更清晰，血管对比度显著提升
- 消融实验验证了CNN与Mamba分支的必要性

## 4. 个人小结
- 创新点：高效融合CNN与Mamba，在医学图像增强任务中取得性能突破
- 局限性：计算复杂度较高，实时部署存在挑战
- 展望：可拓展至其他医学影像增强任务
