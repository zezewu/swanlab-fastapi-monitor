# 🚀 AI Inference Monitoring Service (FastAPI + SwanLab)

这是一个基于 FastAPI 和 PyTorch 构建的图像识别后端推理服务，并创新性地接入了 SwanLab SDK，实现了线上业务请求的实时打点与数据面板监控。

## ✨ 核心特性
* **异步后端：** 使用 FastAPI 处理高并发的 RESTful API 请求。
* **硬件加速：** 自动检测并调用 Mac MPS (Metal Performance Shaders) 进行 PyTorch 推理加速。
* **实时监控：** 将 SwanLab 从离线训练工具魔改为线上监控大盘，实时追踪 API 响应延迟（Latency）和用户多模态请求（Image）。

## 🛠️ 技术栈
* **Backend:** Python, FastAPI, Uvicorn
* **AI/DL:** PyTorch, Torchvision
* **Monitoring:** SwanLab

## 📦 快速启动
1. 安装依赖：`pip install -r requirements.txt`
2. 生成本地测试模型权重：`python save_model.py`
3. 启动后端服务：`python app.py`
4. 访问交互式 API 文档：`http://127.0.0.1:8000/docs`

## 📊 监控看板展示
<img width="631" height="363" alt="image" src="https://github.com/user-attachments/assets/f7c03566-7ed9-439b-bac5-ffe6a6910d37" />

👉 **在线 Demo 看板:** [https://swanlab.cn/space/zezewu]
