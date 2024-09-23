import gradio as gr
import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# 定义函数：获取文章内容
def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article = soup.find('article')
    return article.get_text() if article else soup.get_text()

# 定义函数：生成摘要
def summarize_content(content, model, system_prompt):
    # 创建聊天提示模板
    main_idea_prompt_template = ChatPromptTemplate.from_template(
        f"{system_prompt}\n\n首先提取以下文章的主旨：\n{{content}}\n请给出文章主旨："
    )
    summary_prompt_template = ChatPromptTemplate.from_template(
        f"{system_prompt}\n\n根据以下文章主旨对文章进行摘要，不要丢失重要信息：\n主旨: {{main_idea}}\n文章内容:\n{{content}}\n请给出文章摘要："
    )
    
    # 初始化输出解析器
    output_parser = StrOutputParser()

    # 创建处理链
    main_idea_chain = main_idea_prompt_template | model | output_parser
    summary_chain = summary_prompt_template | model | output_parser

    # 生成文章主旨
    main_idea_response = main_idea_chain.invoke({"content": content})
    main_idea = main_idea_response.strip()

    # 生成文章摘要
    summary_response = summary_chain.invoke({"main_idea": main_idea, "content": content})
    summary = summary_response.strip()

    return summary

# 定义函数：获取文章标题
def get_article_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text() if title_tag else '无标题'

# 定义函数：将摘要保存到文件
def save_to_file(content, title, url):
    # 使用临时文件保存内容
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
        temp_file.write(f"# 摘要\n\n{content}\n\n原文链接: {url}\n标题: {title}")
        temp_file_path = temp_file.name  # 获取临时文件的路径
    return temp_file_path

# 定义 Gradio 界面
def generate_summary(url, system_prompt, model_name):
    # 选择模型
    if model_name.startswith("gpt"):
        model = ChatOpenAI(model=model_name)
    else:
        model = ChatOllama(model=model_name)
    
    # 获取文章内容和标题
    content = get_article_content(url)
    title = get_article_title(url)

    # 生成摘要
    summary = summarize_content(content, model, system_prompt)
    
    # 保存到文件
    file_path = save_to_file(summary, title, url)
    
    return summary, file_path

# 设置 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 文章摘要生成器")

    with gr.Row():
        with gr.Column():
            article_url = gr.Textbox(label='输入文章 URL', value='https://certbot.eff.org/instructions?ws=nginx&os=centosrhel8')
            system_prompt = gr.Textbox(
                label="系统提示词",
                value=(
                    "请根据以下要求进行任务：\n"
                    "1. **提取主旨**：首先提取文章的主旨。\n"
                    "2. **生成摘要**：根据文章主旨对文章进行摘要，不要丢失重要信息。生成的内容需要用中文表示。\n"
                    "3. **代码格式**：如果存在代码或执行语句，以代码段落的形式展示，并识别出代码语言。例如：\n\n"
                    "   ```python\n"
                    "   def example_function():\n"
                    "       print(\"Hello, world!\")\n"
                    "   ```\n\n"
                    "4. **附加信息**：原文链接和标题以引用格式注明在最后。"
                ),
                lines=10
            )
            model_name = gr.Dropdown(
                label='选择模型',
                choices=[
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "gpt-4o-mini",
                    "deepseek-coder-v2:latest",
                    "qwen2:latest",
                    "mistral-nemo:latest",
                    "gemma2:latest",
                    "llama3.1:latest"
                ],
                value="gpt-4o-mini"
            )
            submit_button = gr.Button("生成摘要")
        
        with gr.Column():
            summary_output = gr.Textbox(label="摘要", lines=20, interactive=False)
            file_download = gr.File(label="下载报告")
    
    submit_button.click(generate_summary, [article_url, system_prompt, model_name], [summary_output, file_download])

demo.launch()
