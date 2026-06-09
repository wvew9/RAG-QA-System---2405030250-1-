import os

print("测试1: 检查文档内容")
with open("./documents/自然语言处理简介.txt", "r", encoding="utf-8") as f:
    content = f.read()
print(f"文档内容:\n{content[:200]}...")

print("\n测试2: 检查Ollama连接")
try:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="qwen2:0.5b")
    response = llm.invoke("你好")
    print("LLM测试成功:", response[:50])
except Exception as e:
    print("LLM测试失败:", str(e))

print("\n测试3: 检查嵌入模型")
try:
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vec = embeddings.embed_query("测试")
    print("嵌入测试成功, 向量长度:", len(vec))
except Exception as e:
    print("嵌入测试失败:", str(e))

print("\n测试4: 检查Chroma")
try:
    from langchain_chroma import Chroma
    from langchain.schema import Document
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="./test_chroma", embedding_function=embeddings)
    docs = [Document(page_content="测试文档内容", metadata={"source": "test"})]
    db.add_documents(docs)
    results = db.similarity_search("测试", k=1)
    print("Chroma测试成功, 检索结果:", len(results))
    import shutil
    shutil.rmtree("./test_chroma", ignore_errors=True)
except Exception as e:
    print("Chroma测试失败:", str(e))

print("\n所有测试完成!")