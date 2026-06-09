# RAG 本地知识库问答系统

一个基于 Python 的本地知识库问答系统，使用 Ollama 大模型和 Chroma 向量数据库。

## 项目简介

本项目实现了一个完整的 RAG（检索增强生成）问答系统，支持上传 PDF、DOCX、TXT 文档构建本地知识库，并通过 Streamlit 提供友好的 Web 界面进行问答交互。

## 环境要求与安装步骤

### 1. 安装 Ollama

- 下载并安装 Ollama：https://ollama.com/
- 启动 Ollama 服务
- 下载模型：
  ```bash
  ollama pull deepseek-r1:7b
  ollama pull nomic-embed-text
  ```

### 2. 安装 Python 依赖

- Python 版本：3.9 或更高
- 创建虚拟环境（推荐）：
  ```bash
  python -m venv venv
  # Windows
  venv\Scripts\activate
  # Linux/Mac
  source venv/bin/activate
  ```
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```

## 使用说明

### 运行 Web 应用

```bash
streamlit run app.py
```

### 使用步骤

1. 在侧边栏上传文档（支持 PDF、DOCX、TXT 格式）
2. 点击「构建知识库」按钮处理文档
3. 在问答区域输入问题，点击发送
4. 系统将基于检索到的文档内容给出答案

### 命令行版本

```bash
python rag_chain.py
```

### 测试 Ollama 连接

```bash
python test_ollama.py
```

## 关键技术点说明

### RAG 流程

1. **文档处理**：读取文档 → 文本分块（chunk_size=1000, chunk_overlap=200）
2. **向量化**：使用 nomic-embed-text 模型将文本块转换为向量
3. **存储**：将向量存储在 Chroma 向量数据库中
4. **检索**：根据用户查询，从向量数据库中检索最相关的 3 个文本块
5. **生成**：将检索到的文档作为上下文，使用 deepseek-r1:7b 模型生成答案

### 模型说明

- **大语言模型**：deepseek-r1:7b（可替换为 qwen2:7b 或其他 Ollama 支持的模型）
- **嵌入模型**：nomic-embed-text（也可使用 all-minilm）

### 项目结构

```
RAG-QA-System/
├── app.py              # Streamlit Web 应用
├── rag_chain.py        # 命令行版 RAG 问答系统
├── knowledge_base.py   # 知识库管理模块
├── test_ollama.py      # Ollama 连接测试
├── requirements.txt    # Python 依赖
├── README.md           # 项目说明
├── .gitignore          # Git 忽略配置
├── documents/          # 示例文档目录
└── chroma_db/          # 向量数据库（自动生成）
```

## 项目效果截图

> 提示：运行应用后可以截图放在这里

1. Web 界面主页面
2. 文档上传与知识库构建
3. 问答交互示例

## 本地化打包与部署

### 使用 PyInstaller 打包

创建 `build.spec` 文件：

```python
# -*- mode: python ; coding: utf-8 -*-

import os
import streamlit
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

streamlit_data = collect_data_files('streamlit')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=streamlit_data,
    hiddenimports=[
        'streamlit',
        'streamlit.runtime',
        'chromadb',
        'langchain',
        'langchain_community',
    ] + collect_submodules('streamlit'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RAG-QA-System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

然后运行：

```bash
pyinstaller build.spec
```

打包后的可执行文件位于 `dist/` 目录中。

**注意**：运行打包后的应用仍需要目标机器已安装 Ollama 并下载了所需模型。

## 已知问题与改进方向

- 可以添加更多文档格式支持（如 Markdown、HTML 等）
- 可以实现更高级的检索策略（如重排序、混合检索）
- 可以添加对话历史持久化功能
- 可以优化大模型的提示词模板以获得更好的回答质量

## 许可证

本项目仅供学习使用。
