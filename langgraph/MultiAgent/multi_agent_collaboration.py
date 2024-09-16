import os
import operator
import functools
from langchain_openai import ChatOpenAI
from typing import Annotated, Sequence, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from typing import Literal

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"

# 使用 OpenAI 模型
research_llm =ChatOpenAI(model="gpt-4o-mini")
chart_llm = ChatOpenAI(model="gpt-4o-mini")
table_llm = ChatOpenAI(model="gpt-4o-mini")

# 创建智能体的函数，绑定 LLM（大型语言模型） 和工具
def create_agent(llm, tools, system_message: str):
    """创建一个智能体。"""
    # 定义智能体的提示模板，包含系统消息和工具信息
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",  # 系统消息部分，描述智能体的行为逻辑
                "你是一个有帮助的 AI 助手，正在与其他助手协作。"
                " 使用提供的工具逐步解决问题。"
                " 如果你无法完全回答，没关系，另一个助手会接手，使用不同的工具继续。"
                " 尽你所能执行操作以取得进展。"
                " 如果你或其他助手有最终答案或可交付物，请在回复中加上 FINAL ANSWER 作为前缀，"
                " 让团队知道可以停止操作。"
                " 你可以使用以下工具：{tool_names}。\n{system_message}",  # 提供的工具名称和系统消息
            ),
            MessagesPlaceholder(variable_name="messages"),  # 用于替换的消息占位符
        ]
    )
    
    # 将系统消息部分和工具名称插入到提示模板中
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    # 将提示模板与语言模型和工具绑定
    return prompt | llm.bind_tools(tools)
  
repl = PythonREPL()
# Python REPL 工具，用于执行 Python 代码
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        # 如果代码执行失败，返回错误信息
        return f"Failed to execute. Error: {repr(e)}"
    
    # 成功执行的返回信息，包含执行的代码和标准输出
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    
    # 返回结果，并提示如果任务已完成，请回复 FINAL ANSWER
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
  
  
# 定义 Tavily 搜索工具和 Python 代码执行工具
# Tavily 搜索工具，用于搜索最多 5 条结果
tavily_tool = TavilySearchResults(max_results=5)
# 定义工具列表，包括 Tavily 搜索工具和 Python REPL 工具
tools = [tavily_tool, python_repl]
# 创建工具节点，负责工具的调用
tool_node = ToolNode(tools)



# 定义图中传递的对象，包含消息和发送者信息
class AgentState(TypedDict):
    # messages 是传递的消息，使用 Annotated 和 Sequence 来标记类型
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # sender 是发送消息的智能体
    sender: str
    

# 辅助函数：为智能体创建一个节点
def agent_node(state, agent, name):
    # 调用智能体，获取结果
    result = agent.invoke(state)
    
    # 将智能体的输出转换为适合追加到全局状态的格式
    if isinstance(result, ToolMessage):
        pass  # 如果是工具消息，跳过处理
    else:
        # 将结果转换为 AIMessage，并排除部分字段
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    # 返回更新后的状态，包括消息和发送者
    return {
        "messages": [result],  # 包含新生成的消息
        # 我们使用严格的工作流程，通过记录发送者来知道接下来传递给谁
        "sender": name,
    }
    
# 研究智能体及其节点
research_agent = create_agent(
    research_llm,  # 使用 research_llm 作为研究智能体的语言模型
    [tavily_tool],  # 研究智能体使用 Tavily 搜索工具
    system_message="Before using the search engine, carefully think through and clarify the query. "
    "Then, conduct a single search that addresses all aspects of the query in one go.",  # 系统消息，指导智能体如何使用搜索工具
)
# 使用 functools.partial 创建研究智能体的节点，指定该节点的名称为 "Researcher"
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")


# 图表生成器智能体及其节点
chart_agent = create_agent(
    chart_llm,  # 使用 chart_llm 作为图表生成器智能体的语言模型
    [python_repl],  # 图表生成器智能体使用 Python REPL 工具
    system_message="Create clear and user-friendly charts based on the provided data.",  # 系统消息，指导智能体如何生成图表
)
# 使用 functools.partial 创建图表生成器智能体的节点，指定该节点的名称为 "chart_generator"
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

# 图表生成器智能体及其节点
table_agent = create_agent(
    table_llm,  # 使用 table_llm 作为Table生成器智能体的语言模型
    [python_repl],  # 图表生成器智能体使用 Python REPL 工具
    system_message="Generate a clear and user-friendly table based on the provided data or description. Ensure that the table is organized, properly labeled, and easy to interpret. Handle various data formats, and adapt the table structure to best fit the type of information (e.g., text, numbers, categories).",  # 系统消息，指导智能体如何生成图表
)
# 使用 functools.partial 创建图表生成器智能体的节点，指定该节点的名称为 "table_agent"
table_node = functools.partial(agent_node, agent=chart_agent, name="table_agent")


# 路由器函数，用于决定下一步是执行工具还是结束任务
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]  # 获取当前状态中的消息列表
    last_message = messages[-1]  # 获取最新的一条消息
    
    # 如果最新消息包含工具调用，则返回 "call_tool"，指示执行工具
    if last_message.tool_calls:
        return "call_tool"
    
    # 如果最新消息中包含 "FINAL ANSWER"，表示任务已完成，返回 "__end__" 结束工作流
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    
    # 如果既没有工具调用也没有完成任务，继续流程，返回 "continue"
    return "continue"


# 创建一个状态图 workflow，使用 AgentState 来管理状态
workflow = StateGraph(AgentState)

# 将研究智能体节点、图表生成器智能体节点和工具节点添加到状态图中
workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("table_generator", table_node)
workflow.add_node("call_tool", tool_node)


# 为 "Researcher" 智能体节点添加条件边，根据 router 函数的返回值进行分支
workflow.add_conditional_edges(
    "Researcher",
    router,  # 路由器函数决定下一步
    {
        "continue": "table_generator",  # 如果 router 返回 "continue"，则传递到 chart_generator
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "chart_generator" 智能体节点添加条件边
workflow.add_conditional_edges(
    "chart_generator",
    router,  # 同样使用 router 函数决定下一步
    {
        "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

workflow.add_conditional_edges(
    "table_generator",
    router,  # 同样使用 router 函数决定下一步
    {
        "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)


# 为 "call_tool" 工具节点添加条件边，基于“sender”字段决定下一个节点
# 工具调用节点不更新 sender 字段，这意味着边将返回给调用工具的智能体
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],  # 根据 sender 字段判断调用工具的是哪个智能体
    {
        "Researcher": "Researcher",  # 如果 sender 是 Researcher，则返回给 Researcher
        "chart_generator": "chart_generator",  # 如果 sender 是 chart_generator，则返回给 chart_generator
    },
)


# 添加开始节点，将流程从 START 节点连接到 Researcher 节点
workflow.add_edge(START, "Researcher")

# 编译状态图以便后续使用
graph = workflow.compile()


# 可视化图
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception as e:
#     print(f"Error generating graph: {e}")
    
    
    
# events = graph.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Obtain the GDP of the United States from 2010 to 2020, "
#             "and then plot a line chart with Python. End the task after generating the chart"
#             )
#         ],
#     },
#     # 设置最大递归限制
#     {"recursion_limit": 20},
#     stream_mode="values"
# )

# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()  # 打印消息内容
