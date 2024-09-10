import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from graphviz import Source

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage

import json
from langchain_core.messages import ToolMessage

# 定义 BasicToolNode，用于执行工具请求
class BasicToolNode:
    """一个在最后一条 AIMessage 中执行工具请求的节点。
    
    该节点会检查最后一条 AI 消息中的工具调用请求，并依次执行这些工具调用。
    """

    def __init__(self, tools: list) -> None:
        # tools 是一个包含所有可用工具的列表，我们将其转化为字典，
        # 通过工具名称（tool.name）来访问具体的工具
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """执行工具调用
        
        参数:
        inputs: 包含 "messages" 键的字典，"messages" 是对话消息的列表，
                其中最后一条消息可能包含工具调用的请求。
        
        返回:
        包含工具调用结果的消息列表
        """
        # 获取消息列表中的最后一条消息，判断是否包含工具调用请求
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("输入中未找到消息")

        # 用于保存工具调用的结果
        outputs = []

        # 遍历工具调用请求，执行工具并将结果返回
        for tool_call in message.tool_calls:
            # 根据工具名称找到相应的工具，并调用工具的 invoke 方法执行工具
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            # 将工具调用结果作为 ToolMessage 保存下来
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),  # 工具调用的结果以 JSON 格式保存
                    name=tool_call["name"],  # 工具的名称
                    tool_call_id=tool_call["id"],  # 工具调用的唯一标识符
                )
            )
        # 返回包含工具调用结果的消息
        return {"messages": outputs}

# 环境配置
def setup_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph ChatBot"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_api_base = os.getenv('OPENAI_API_BASE')
    return openai_api_key, openai_api_base

# 定义状态类型
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 路由工具调用函数
def route_tools(state: State) -> Literal["tools", "__end__"]:
    try:
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"输入状态中未找到消息: {state}")

        print(f"AI message: {ai_message}")  # 打印 AI 消息

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            print("Tool calls found, routing to tools...")
            return "tools"
        print("No tool calls, ending session.")
        return "__end__"
    except Exception as e:
        raise RuntimeError(f"工具路由失败: {e}")

# 初始化模型和工具
def initialize_model_and_tools(api_key, api_base):
    print(f"TAVILY_API_KEY: {os.getenv('TAVILY_API_KEY')}")
    tool = TavilySearchResults(max_results=2)
    result = tool.invoke(("What's a 'node' in LangGraph?"))
    print(f"Tool result: {result}")  # 打印工具结果
    tools = [tool]
    chat_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, openai_api_base=api_base)
    llm_with_tools = chat_model.bind_tools(tools)
    
    print(f"Available tools: {tools}")
    
    # print(f"Bound tools: {llm_with_tools.tools}")  # 打印绑定的工具
    return llm_with_tools, tools

# 聊天机器人节点函数
def chatbot(state: State, llm_with_tools):
    try:
        messages = state["messages"]
        print(f"Input messages: {messages}")  # 打印输入的消息
        response = llm_with_tools.invoke(messages)
        print(f"Model response: {response}")  # 打印模型的响应
        return {"messages": [response]}
    except Exception as e:
        raise RuntimeError(f"调用聊天模型失败: {e}")

# 构建状态图
def build_graph(llm_with_tools, tools):
    graph_builder = StateGraph(State)

    # 添加聊天机器人节点
    graph_builder.add_node("chatbot", lambda state: chatbot(state, llm_with_tools))

    # 添加工具节点
    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # 添加条件边，判断是否需要调用工具
    graph_builder.add_conditional_edges(
        "chatbot", 
        route_tools, 
        {"tools": "tools", "__end__": "__end__"}
    )

    # 工具调用结束后继续对话
    graph_builder.add_edge("tools", "chatbot")

    # 从 START 节点开始对话
    graph_builder.add_edge(START, "chatbot")
    
    return graph_builder.compile()

# 显示状态图
def display_graph(graph):
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))

    except Exception as e:
        print(f"无法显示图形: {e}")

# def display_graph(graph):
#     try:
#         # 使用 draw_mermaid() 方法获取 Mermaid 图的文本表示
#         mermaid_code = graph.get_graph().draw_mermaid()

#         # 使用 graphviz 将 mermaid_code 转换为图像
#         graphviz_source = Source(mermaid_code)
#         graphviz_source.render('graph_output', format='png')

#         # 使用 matplotlib 读取和显示生成的 PNG 图片
#         img = mpimg.imread('graph_output.png')
#         plt.imshow(img)
#         plt.axis('off')  # 隐藏坐标轴
#         plt.show()

#     except Exception as e:
#         print(f"无法显示图形: {e}")

# 主对话循环
def main():
    api_key, api_base = setup_environment()
    
    llm_with_tools, tools = initialize_model_and_tools(api_key, api_base)
    graph = build_graph(llm_with_tools, tools)

    # 可选：显示状态图
    # display_graph(graph)

    # 对话循环
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        try:
            # 处理用户输入并生成回复
            for event in graph.stream({"messages": [("user", user_input)]}):
                for value in event.values():
                    if isinstance(value["messages"][-1], BaseMessage):
                        print("Assistant:", value["messages"][-1].content)
        except Exception as e:
            print(f"对话处理失败: {e}")

if __name__ == "__main__":
    main()
