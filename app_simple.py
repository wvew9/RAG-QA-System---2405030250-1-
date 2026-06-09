import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(
    page_title="RAG 本地知识库问答系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG 本地知识库问答系统")

def init_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

init_session_state()

sidebar = st.sidebar
with sidebar:
    st.header("📚 知识库管理")
    
    uploaded_files = st.file_uploader(
        "上传文档 (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="支持 PDF、DOCX 和 TXT 格式"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"已选择 {len(uploaded_files)} 个文件")
    
    if st.button("🔨 构建知识库", type="primary"):
        if st.session_state.uploaded_files:
            with st.spinner("正在处理文档..."):
                success_count = 0
                error_count = 0
                for uploaded_file in st.session_state.uploaded_files:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            text = read_document(tmp_file_path)
                            st.session_state.processed_files.append({
                                "name": uploaded_file.name,
                                "size": len(text),
                                "chunks": len(text) // 1000 + 1
                            })
                            success_count += 1
                        finally:
                            os.unlink(tmp_file_path)
                    except Exception as e:
                        error_count += 1
                        st.error(f"处理 {uploaded_file.name} 时出错: {str(e)}")
                
                if success_count > 0:
                    st.success(f"知识库构建完成！成功处理 {success_count} 个文件")
                if error_count > 0:
                    st.warning(f"有 {error_count} 个文件处理失败")
                st.session_state.uploaded_files = []
        else:
            st.warning("请先上传文档")
    
    st.divider()
    
    st.subheader("📊 知识库状态")
    st.metric("已处理文件数", len(st.session_state.processed_files))
    
    if st.session_state.processed_files:
        st.write("已处理的文件:")
        for file in st.session_state.processed_files:
            st.write(f"- {file['name']} (约 {file['chunks']} 个文本块)")
    
    st.divider()
    
    if st.button("🗑️ 清空数据"):
        st.session_state.processed_files = []
        st.session_state.uploaded_files = []
        st.success("数据已清空")

st.subheader("💬 问答交互")
st.info("请先上传文档并构建知识库，然后开始提问")

if question := st.chat_input("请输入您的问题..."):
    if not st.session_state.processed_files:
        st.warning("请先上传文档并构建知识库")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write("文档中未找到相关答案")