import bs4
import tiktoken
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import chromadb
from langchain.schema import Document

# 配置 API Key 和 Base URL
openai_api_key = os.getenv('OPENAI_API_KEY')  # 将密钥从环境变量中读取
openai_api_base = os.getenv('OPENAI_API_BASE')

# 定义加载 PDF 内容的函数
def load_pdf_content(file_path):
    loader = PyPDFium2Loader(file_path)
    return loader.load()

# 定义分割文本的函数
def split_text(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return text_splitter.split_documents(docs)

# 定义估算 Token 数量的函数
def estimate_tokens(docs, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(doc.page_content)) for doc in docs)
    return total_tokens

# 定义创建向量存储的函数
def create_vector_store(splits):
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_model, 
        persist_directory="./chroma_langchain_db"
    )
    
    # 持久化存储
    return vector_store
    
def get_vector_store():
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    
    vector_store_from_client = Chroma(
        persist_directory="./chroma_langchain_db",  # 确保与保存路径一致
        embedding_function=embedding_model,
    )
    return vector_store_from_client



# 定义创建 RAG 链的函数
def create_rag_chain(retriever, llm, prompt_template):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
def create_rag_chain2(retriever, llm, prompt_template):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

# -------------------- 方法定义结束，以下为调用部分 --------------------

if __name__ == "__main__":
    # # 1. 加载 PDF 内容
    # pages = load_pdf_content("docs/pdf/存在主义心理治疗.pdf")

    # # 打印第一页的前20个字符
    # print(pages[0].page_content[:20])

    # # 2. 分割文本
    # all_splits = split_text(pages)
    # print(f"Total splits: {len(all_splits)}")  # 打印分割后的文档块数量

    # # 3. 估算 token 数量
    # tokens = estimate_tokens(all_splits)
    # print(f"Total token count in docs: {tokens}")

    # # 4. 创建向量存储
    # vectorstore = create_vector_store(all_splits)
    
    vectorstore = get_vector_store()
    print(f"Number of stored documents: {vectorstore._collection.count()}")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # 5. 执行检索任务
    query = " ?"
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs[0].page_content)  # 打印第一个检索到的文档内容

    # 6. 初始化 RAG 链
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    prompt = hub.pull("rlm/rag-prompt")
    print(prompt.messages)
    
    rag_chain = create_rag_chain(retriever, llm, prompt)
    print("=============\nRAG chain ready.\n")
    
    # 7. rag_chain 的定义和初始化已经完成
    question_batch = [
        "什么是存在主义心理治疗?",
        "个体如何面对生命的终结性和无意义感？?",
        "怎么样寻找个人意义？?",
    ]

    # for i, question in enumerate(question_batch):
    #     print(f"Question: {question}")
    #     print(f"Answer: {rag_chain.invoke(question)}\n")

    # # 8. 测试 RAG 链的批量处理能力
    messages = rag_chain.batch(question_batch)
    print("RAG chain processing complete.\n")    
    for i, message in enumerate(messages):
        print(f"Question: {question_batch[i]}")
        print(f"Answer: {message}\n")



    # 在 LangChain Hub 上找一个可用的 RAG 提示词模板，测试对比两者的召回率和生成质量。
    # prompt = hub.pull("rlm/rag-prompt-llama")
    # print(prompt.messages)
    
    # rag_chain = create_rag_chain(retriever, llm, prompt)
    # print("=============\nRAG chain ready.\n")
    
    # messages = rag_chain.batch(question_batch)
    # print("RAG chain processing complete.\n")    
    # for i, message in enumerate(messages):
    #     print(f"Question: {question_batch[i]}")
    #     print(f"Answer: {message}\n")


    # 定义RAG链
    def create_custom_rag_chain(retriever, llm, prompt_template):
        # 重新设计提示词模板
        def format_docs(docs):
            return """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            尽量保持书中原文。
            Question: {question} 
            Context: {context} 
            Answer:
            """

        # 确保 retriever、prompt_template 和 llm 是可组合的可执行步骤
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

    rag_chain = create_custom_rag_chain(retriever, llm, prompt)
    print("=============\nRAG chain ready.\n")
    
    messages = rag_chain.batch(question_batch)
    print("RAG chain processing complete.\n")    
    for i, message in enumerate(messages):
        print(f"Question: {question_batch[i]}")
        print(f"Answer: {message}\n")
