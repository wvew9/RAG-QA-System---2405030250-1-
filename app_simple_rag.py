import streamlit as st
import os
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="RAG 本地知识库问答系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG 本地知识库问答系统")

def init_session_state():
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = []
    if "chunk_metadata" not in st.session_state:
        st.session_state.chunk_metadata = []

def read_pdf(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file_path):
    from docx import Document
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

def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks

def get_embedding(text):
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings.embed_query(text)

def retrieve_relevant_chunks(query, k=3):
    if not st.session_state.chunk_embeddings:
        return []
    
    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    similarities = []
    for i, chunk_embedding in enumerate(st.session_state.chunk_embeddings):
        chunk_embedding = np.array(chunk_embedding).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
        similarities.append((i, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for i, similarity in similarities[:k]:
        results.append({
            "content": st.session_state.text_chunks[i],
            "metadata": st.session_state.chunk_metadata[i],
            "similarity": similarity
        })
    
    return results

def generate_answer(query, context_docs):
    from langchain_ollama import OllamaLLM
    
    context = "\n\n".join([doc["content"] for doc in context_docs])
    
    prompt = f"""基于以下参考文档回答问题：

参考文档：
{context}

用户问题：{query}

如果文档中有相关信息，请基于文档内容回答。如果没有相关信息，请明确说"文档中未找到相关答案"。

回答："""
    
    llm = OllamaLLM(model="qwen2:0.5b")
    response = llm.invoke(prompt)
    return response, context_docs

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
    
    use_sample_docs = st.checkbox("使用示例文档", value=True)
    
    if st.button("🔨 构建知识库", type="primary"):
        with st.spinner("正在处理文档..."):
            st.session_state.text_chunks = []
            st.session_state.chunk_embeddings = []
            st.session_state.chunk_metadata = []
            
            documents = []
            
            if use_sample_docs:
                docs_folder = "./documents"
                if os.path.exists(docs_folder):
                    for filename in os.listdir(docs_folder):
                        file_path = os.path.join(docs_folder, filename)
                        if os.path.isfile(file_path):
                            ext = os.path.splitext(filename)[1].lower()
                            if ext in [".pdf", ".docx", ".txt"]:
                                try:
                                    text = read_document(file_path)
                                    documents.append({"name": filename, "content": text})
                                except:
                                    pass
                    st.info(f"已加载 {len(documents)} 个示例文档")
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        text = read_document(tmp_file_path)
                        documents.append({"name": uploaded_file.name, "content": text})
                        st.success(f"已处理: {uploaded_file.name}")
                    finally:
                        os.unlink(tmp_file_path)
            
            if documents:
                try:
                    for doc in documents:
                        chunks = split_text(doc["content"])
                        for i, chunk in enumerate(chunks):
                            embedding = get_embedding(chunk)
                            st.session_state.text_chunks.append(chunk)
                            st.session_state.chunk_embeddings.append(embedding)
                            st.session_state.chunk_metadata.append({
                                "source": doc["name"],
                                "chunk": i
                            })
                    st.success(f"✅ 知识库构建完成！共 {len(st.session_state.text_chunks)} 个文本块")
                except Exception as e:
                    st.error(f"构建知识库失败: {str(e)}")
            else:
                st.warning("没有文档需要处理")
    
    st.divider()
    
    st.subheader("📊 知识库状态")
    st.metric("文本块数量", len(st.session_state.text_chunks))
    
    if st.button("🗑️ 清空对话历史"):
        st.session_state.chat_history = []
        st.success("对话历史已清空")

st.subheader("💬 问答交互")

for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

if question := st.chat_input("请输入您的问题..."):
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        if not st.session_state.text_chunks:
            st.warning("⚠️ 知识库尚未构建，请先上传文档并点击'构建知识库'")
        else:
            try:
                with st.spinner("检索相关文档..."):
                    context_docs = retrieve_relevant_chunks(question, k=3)
                
                if not context_docs:
                    st.write("文档中未找到相关答案")
                else:
                    with st.spinner("生成回答..."):
                        answer, sources = generate_answer(question, context_docs)
                    
                    st.write(answer)
                    
                    with st.expander("📖 查看参考来源"):
                        for i, doc in enumerate(sources):
                            st.write(f"**来源**: {doc['metadata']['source']}")
                            st.write(doc["content"][:200] + "...")
                            st.divider()
            
            except Exception as e:
                st.error(f"回答时出错: {str(e)}")
                answer = "抱歉，回答过程中出现错误"
    
    st.session_state.chat_history.append((question, answer))