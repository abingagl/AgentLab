# 导入相关模块，包括运算符、输出解析器、聊天模板、ChatOpenAI 和 运行器
from operator import itemgetter
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 配置 API Key 和 Base URL
openai_api_key = os.getenv('OPENAI_API_KEY')  # 将密钥从环境变量中读取
openai_api_base = os.getenv('OPENAI_API_BASE')

llm = ChatOpenAI(model="gpt-4o-mini",openai_api_key = openai_api_key, openai_api_base = openai_api_base)

# 创建一个计划器，生成一个关于给定输入的论证
planner = (
    ChatPromptTemplate.from_template("生成关于以下内容的论点: {input}")
    | llm
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

# 创建正面论证的处理链，列出关于基础回应的正面或有利的方面
arguments_for = (
    ChatPromptTemplate.from_template(
        "列出关于{base_response}的正面或有利的方面"
    )
    | llm
    | StrOutputParser()
)

# 创建反面论证的处理链，列出关于基础回应的反面或不利的方面
arguments_against = (
    ChatPromptTemplate.from_template(
        "列出关于{base_response}的反面或不利的方面"
    )
    | llm
    | StrOutputParser()
)

# 创建最终响应者，综合原始回应和正反论点生成最终的回应
final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "正面观点:\n{results_1}\n\n反面观点:\n{results_2}"),
            ("system", "给出批评后生成最终回应"),
        ]
    )
    | llm
    | StrOutputParser()
)

# 构建完整的处理链，从生成论点到列出正反论点，再到生成最终回应
chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)


print(chain.invoke({"input": "房地产低迷"}))