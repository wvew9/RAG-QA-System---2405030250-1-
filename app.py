import streamlit as st
import os
import tempfile
from knowledge_base import KnowledgeBase

st.set_page_config(
    page_title="RAG 本地知识库问答系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG 本地知识库问答系统")

def init_session_state():
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "upload_error" not in st.session_state:
        st.session_state.upload_error = ""

def create_qa_chain():
    from langchain.chains import ConversationalRetrievalChain
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    
    prompt_template = """你是一个专业的问答助手。请基于以下参考文档回答用户的问题。

参考文档：
{context}

对话历史：
{chat_history}

用户问题：{question}

请基于提供的参考文档回答问题。如果参考文档中没有相关信息，请明确说"文档中未找到相关答案"。

回答："""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    vectorstore = st.session_state.kb.get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="qwen2:0.5b")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

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
        
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
        st.session_state.upload_error = ""
    
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
                            st.session_state.kb.add_document(tmp_file_path)
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
                st.session_state.qa_chain = None
                st.rerun()
        else:
            st.warning("请先上传文档")
    
    st.divider()
    
    st.subheader("📊 知识库状态")
    doc_count = st.session_state.kb.get_document_count()
    sources = st.session_state.kb.get_unique_sources()
    
    st.metric("文本块数量", doc_count)
    st.write(f"文档来源 ({len(sources)}):")
    for source in sources:
        st.write(f"- {source}")
    
    st.divider()
    
    if st.button("🗑️ 清空对话历史"):
        st.session_state.chat_history = []
        st.success("对话历史已清空")
        st.rerun()

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
        if st.session_state.kb.get_document_count() == 0:
            st.warning("知识库为空，请先上传文档并构建知识库")
        else:
            with st.spinner("正在初始化问答系统..."):
                if st.session_state.qa_chain is None:
                    st.session_state.qa_chain = create_qa_chain()
        
            with st.spinner("正在思考..."):
                try:
                    result = st.session_state.qa_chain({
                        "question": question,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = result["answer"]
                    source_docs = result["source_documents"]
                    
                    st.write(answer)
                    
                    if source_docs:
                        with st.expander("📖 查看参考来源"):
                            for i, doc in enumerate(source_docs):
                                st.write(f"**来源 {i+1}**: {doc.metadata['source']}")
                                st.write(doc.page_content[:200] + "...")
                                st.divider()
                except Exception as e:
                    st.error(f"回答时出错: {str(e)}")
                    answer = "抱歉，回答过程中出现错误"
    
    st.session_state.chat_history.append((question, answer))