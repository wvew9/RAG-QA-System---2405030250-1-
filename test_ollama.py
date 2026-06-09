from langchain_ollama import OllamaLLM

def test_ollama_connection():
    print("正在测试Ollama连接...")
    try:
        llm = OllamaLLM(model="deepseek-r1:7b")
        response = llm.invoke("你好，请简单介绍一下自己。")
        print("Ollama API测试成功！")
        print("模型回复：", response)
        return True
    except Exception as e:
        print("Ollama API测试失败：", str(e))
        print("请确保：")
        print("1. Ollama已安装并正在运行")
        print("2. deepseek-r1:7b模型已下载")
        print("3. 如果使用其他模型，请修改代码中的模型名称")
        return False

if __name__ == "__main__":
    test_ollama_connection()