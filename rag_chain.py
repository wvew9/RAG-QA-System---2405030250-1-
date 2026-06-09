from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from knowledge_base import KnowledgeBase

class RAGQASystem:
    def __init__(self, model_name="deepseek-r1:7b", embedding_model="nomic-embed-text"):
        self.kb = KnowledgeBase(model_name=embedding_model)
        self.llm = Ollama(model=model_name)
        self.chat_history = []
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self):
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

        vectorstore = self.kb.get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def ask(self, question):
        result = self.qa_chain({
            "question": question,
            "chat_history": self.chat_history
        })
        
        self.chat_history.append((question, result["answer"]))
        
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"]
        }
    
    def clear_history(self):
        self.chat_history = []
    
    def get_knowledge_base(self):
        return self.kb

if __name__ == "__main__":
    print("=== RAG 问答系统 ===")
    print("正在初始化系统...")
    
    qa_system = RAGQASystem()
    
    docs_folder = "./documents"
    import os
    if os.path.exists(docs_folder):
        print(f"\n正在从 {docs_folder} 加载文档...")
        qa_system.get_knowledge_base().add_documents_from_folder(docs_folder)
    else:
        print(f"\n文件夹 {docs_folder} 不存在，跳过文档加载")
    
    print("\n知识库状态：")
    print(f"文本块数量: {qa_system.get_knowledge_base().get_document_count()}")
    print(f"文档来源: {qa_system.get_knowledge_base().get_unique_sources()}")
    
    print("\n=== 开始问答 ===")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史\n")
    
    while True:
        question = input("你: ")
        
        if question.lower() in ["quit", "exit"]:
            print("再见！")
            break
        
        if question.lower() == "clear":
            qa_system.clear_history()
            print("对话历史已清空\n")
            continue
        
        if not question.strip():
            continue
        
        print("\n正在思考...")
        result = qa_system.ask(question)
        
        print(f"\n助手: {result['answer']}\n")
        
        print("参考来源：")
        for i, doc in enumerate(result["source_documents"]):
            print(f"  [{i+1}] {doc.metadata['source']}")
        print()
