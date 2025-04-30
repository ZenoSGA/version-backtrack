# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:22:11 2025

@author: 21164
"""


import streamlit as st
import os
import json
import re
import tempfile
import fitz  # PyMuPDF
from openai import OpenAI
import plotly.graph_objects as go
from fpdf import FPDF
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from markmap2 import markmap_auto_export

# 初始化 DeepSeek 客户端

def init_deepseek_client():
    try:
        client = OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
        return client
    except Exception as e:
        st.error(f"DeepSeek客户端初始化失败：{str(e)}")
        return None

# 初始化本地模型客户端（仅用于专家建议）

def init_local_model_client():
    try:
        client = OpenAI(
            api_key="local_model",  # 占位值，本地模型通常不验证密钥
            base_url="http://localhost:8000/v1"  # 本地模型的 API 端点
        )
        return client
    except Exception as e:
        st.error(f"本地模型客户端初始化失败：{str(e)}")
        return None

def search_related_papers(deepseek_client, text):
    keywords = extract_keywords_via_deepseek(deepseek_client, text)
    search_query = " OR ".join(keywords) 
    serpapi_key = st.secrets["SERPAPI_API_KEY"]
    url = f"https://serpapi.com/search?engine=google_scholar&q={search_query}&api_key={serpapi_key}&num=10"
    response = requests.get(url)
    data = response.json()
    
    papers = []
    for result in data.get("organic_results", []):
        summary = result.get("snippet", "").lower()
        match_count = sum(1 for kw in keywords if kw.lower() in summary)  # 计算匹配关键词数量
        paper = {
            "title": result.get("title"),
            "summary": summary,
            "match_count": match_count
        }
        papers.append(paper)
    
    # 按匹配度排序，取前3篇
    papers = sorted(papers, key=lambda x: x["match_count"], reverse=True)[:3]
    return papers


# PDF处理模块
def extract_text_from_pdf(uploaded_file):
    """提取PDF文本内容（带目录结构）"""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        toc = doc.get_toc()
        if toc:
            text += "# 论文目录\n"
            for level, title, _ in toc:
                text += f"{'#'*(level+1)} {title}\n"
            text += "\n"
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF解析失败：{str(e)}")
        return ""

def create_pdf_report(result):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(current_dir, "fonts")
    class PDF(FPDF):
        def header(self):
            self.set_font("MicrosoftYaHei", "B", 16)
            self.cell(0, 10, "学位论文智能评审报告", 0, 1, "C")
        
        def chapter_title(self, title):
            self.set_font("MicrosoftYaHei", "B", 12)
            self.cell(0, 10, title, 0, 1)
            self.ln(4)
            
        def chapter_body(self, body):
            self.set_font("MicrosoftYaHei", "", 12)
            self.multi_cell(0, 10, body)
            self.ln()
    
    pdf = PDF()
    try:
        pdf.add_font("MicrosoftYaHei", "", os.path.join(font_dir, "msyh.ttf"), uni=True)
        pdf.add_font("MicrosoftYaHei", "B", os.path.join(font_dir, "msyhbd.ttf"), uni=True)
    except Exception as e:
        st.error(f"字体加载失败：{str(e)}")
        st.stop()
    pdf.add_page()
    pdf.set_font("MicrosoftYaHei", "B", 16)
    pdf.cell(0, 40, "学位论文智能评审报告", 0, 1, "C")
    pdf.ln(20)
    pdf.set_font("MicrosoftYaHei", "", 12)
    pdf.cell(0, 10, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    
    pdf.add_page()
    pdf.chapter_title("评分标准说明")
    for dim, criteria in DIMENSION_CRITERIA.items():
        pdf.set_font("MicrosoftYaHei", "B", 12)
        pdf.cell(0, 10, f"{dim}: {criteria['description']}", 0, 1)
        pdf.set_font("MicrosoftYaHei", "", 10)
        for score, desc in criteria['criteria'].items():
            pdf.multi_cell(0, 8, f"{score}: {desc}")
        pdf.ln(5)
    
    pdf.add_page()
    pdf.chapter_title("分项详细评述")
    for dim in ["工作量", "创新性", "规范性", "严谨性", "深度"]:
        pdf.chapter_title(dim)
        comment = result["comments"][dim]
        pdf.chapter_body(f"优点：{comment.get('strengths', '模型未提供优点')}")
        pdf.chapter_body(f"不足：{comment.get('weaknesses', '模型未提供不足')}")
        pdf.chapter_body(f"建议：{comment.get('suggestions', '模型未提供建议')}")
        pdf.ln()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# 摘要和关键词提取（使用 DeepSeek）
def generate_summary_via_deepseek(deepseek_client, text, mode="青年科学基金"):
    prompts = {
        # ...保留原有其他模式...
        "青年科学基金": """请按照青年科学基金摘要格式生成结构化内容，要求用10句话严格按以下结构组织：
1. 背景
（1）社会上的较宽泛问题：[一句话]
（2）具体困扰：[一句话]
（3）已解决/未解决问题：[一句话]
（4）本文研究内容：[一句话]
2. 研究方法：[一句话]
3. 研究工作：[一句话]
4. 解决问题：[一句话]
5. 研究结论：[一句话]
6. 实际作用：[一句话]
7. 科学意义：[一句话]

请用中文分点回答，每个编号/字母后直接跟内容，不要空行，不要使用markdown格式。"""
    }
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位精通国家自然科学基金申报的科研专家"},
                {"role": "user", "content": f"{prompts[mode]}\n\n论文内容：{text[:30000]}"}
            ],
            temperature=0.2,  # 降低随机性
            max_tokens=500    # 增加token限额
        )
        return _format_output(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"摘要生成失败：{str(e)}")
        return "摘要生成失败，请重试"
def _format_output(text):
    # 后处理确保格式规范
    formatted = []
    for line in text.split('\n'):
        line = line.replace("（1）", "（1）").replace("（2）", "（2）").replace("（3）", "（3）")
        line = line.replace("1.", "\n1.").replace("2.", "\n2.").replace("7.", "\n7.")
        formatted.append(line.strip())
    return "\n".join(formatted).strip()

def extract_keywords_via_deepseek(deepseek_client, text):
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的学术论文关键词提取器"},
                {"role": "user", "content": f"请从以下论文内容中提取5个最重要的关键词，用中文逗号分隔：\n\n{text[:30000]}"}
            ],
            temperature=0.1,
            max_tokens=100
        )
        keywords = response.choices[0].message.content.strip()
        keywords = re.sub(r"[^\w\u4e00-\u9fa5,，、]", "", keywords)
        keywords = [kw.strip() for kw in re.split("[,，、]", keywords) if kw.strip()]
        return keywords[:5]
    except Exception as e:
        st.error(f"关键词提取失败：{str(e)}")
        return ["关键词提取失败"]

# 评估维度定义
DIMENSION_CRITERIA = {
    "工作量": {
        "description": "评估论文研究工作的数量、广度和完成度",
        "criteria": {
            "5分": "工作量非常饱满，包含大量实验/分析/研究，远超同类论文平均水平",
            "4分": "工作量饱满，涵盖研究所需的主要内容，达到或略超平均水平",
            "3分": "工作量基本满足要求，但某些方面可以进一步扩展",
            "2分": "工作量明显不足，关键部分缺失或过于简略",
            "1分": "工作量严重不足，无法支撑论文结论"
        }
    },
    "创新性": {
        "description": "评估论文的创新程度和原创性贡献",
        "criteria": {
            "5分": "具有重大理论或实践创新，开辟新研究方向",
            "4分": "有明显创新点，对现有方法有实质性改进",
            "3分": "有一定创新性，但创新程度有限",
            "2分": "创新性不足，主要是已有工作的简单组合",
            "1分": "基本没有创新性，完全重复已有工作"
        }
    },
    "规范性": {
        "description": "评估论文写作和格式的规范性",
        "criteria": {
            "5分": "写作极其规范，结构严谨，格式完美符合学术规范",
            "4分": "写作规范，结构合理，仅有少量格式问题",
            "3分": "基本规范，但存在一些结构或格式问题",
            "2分": "规范性较差，存在多处明显问题",
            "1分": "极不规范，严重影响阅读和理解"
        }
    },
    "严谨性": {
        "description": "评估论文研究方法和论证的严谨性",
        "criteria": {
            "5分": "方法设计极其严谨，论证充分，无逻辑漏洞",
            "4分": "方法设计合理，论证基本充分，无明显漏洞",
            "3分": "方法基本合理，但某些论证不够充分",
            "2分": "方法设计存在明显缺陷，论证不充分",
            "1分": "方法设计严重缺陷，论证极不严谨"
        }
    },
    "深度": {
        "description": "评估论文研究的理论深度和分析深度",
        "criteria": {
            "5分": "研究极具深度，理论分析透彻，见解深刻",
            "4分": "研究有深度，分析较为深入",
            "3分": "研究有一定深度，但某些方面分析不够深入",
            "2分": "研究深度不足，分析较为表面",
            "1分": "研究极其肤浅，缺乏深入分析"
        }
    }
}

# 增强版分析模块（DeepSeek + 本地模型）
def generate_summary_and_keypoints(deepseek_client, paper_text, dim, criteria):
    prompt = f"""
    请你作为一位学术论文分析专家，针对"{dim}"维度，从以下论文内容中提取与该维度相关的关键信息和重点内容。"{dim}"的定义是：{criteria['description']}

    请提供一份简洁的总结，突出与"{dim}"维度相关的主要观点、方法、实验结果等，不少于200字。

    论文内容如下：
    {paper_text[:30000]}
    """
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位学术论文分析专家，擅长从论文中提取关键信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"DeepSeek 总结和重点提取失败：{str(e)}")
        return "总结和重点提取失败"

def get_expert_suggestions(local_client, summary_and_keypoints, dim):
    prompt = f"""
    你是一位资深的论文评审专家，请基于以下针对"{dim}"维度的论文总结和重点内容，提供具体的改进建议或肯定性评价。

    总结和重点内容：
    {summary_and_keypoints}

    请给出针对性的建议，如果该维度表现良好，请说明"该维度表现优秀，暂无明显改进建议"，不少于150字。
    """
    try:
        response = local_client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "你是资深的论文评审专家，擅长提供建设性建议。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"专家建议获取失败：{str(e)}")
        return "专家建议获取失败"

def generate_final_comments(deepseek_client, paper_text, expert_suggestions, dim):
    prompt = f"""
    你是一位专业的学术论文评审专家，请基于你对论文的理解和专家的建议，对论文在"{dim}"维度进行综合评价。

    论文内容：
    {paper_text[:30000]}

    专家建议：
    {expert_suggestions}

    请从以下方面进行评价，并给出0-5分的评分（保留一位小数）：
    1. 优点（详细展开，不少于150字）(说中文)
    2. 不足（指出存在的问题，不少于150字）(说中文)
    3. 建议（结合专家建议和自己的理解，不少于150字）(说中文)

    输出格式为 JSON：
    {{
        "score": <你的评分>,
        "strengths": "...",
        "weaknesses": "...",
        "suggestions": "..."
    }}
    """
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的学术论文评审专家，擅长综合评价论文。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"最终评论生成失败：{str(e)}")
        return {"score": 0, "strengths": "生成失败", "weaknesses": "生成失败", "suggestions": "生成失败"}

def generate_detailed_rating_via_deepseek_and_local_model(deepseek_client, local_client, paper_text):
    result = {"scores": {}, "comments": {}}
    for dim, criteria in DIMENSION_CRITERIA.items():
        with st.spinner(f"正在分析维度：{dim}"):
            # Step 1: DeepSeek 总结和重点提取
            summary_and_keypoints = generate_summary_and_keypoints(deepseek_client, paper_text, dim, criteria)
            # Step 2: 专家模型提供建议
            expert_suggestions = get_expert_suggestions(local_client, summary_and_keypoints, dim)
            # Step 3: DeepSeek 生成最终评论和评分
            final_result = generate_final_comments(deepseek_client, paper_text, expert_suggestions, dim)
            result["scores"][dim] = {"value": final_result["score"], "reason": "基于综合评价"}
            result["comments"][dim] = {
                "strengths": final_result["strengths"],
                "weaknesses": final_result["weaknesses"],
                "suggestions": final_result["suggestions"]
            }
    return result

# 思维导图生成模块（使用 DeepSeek）
def ask_deepseek_for_markmap(deepseek_client, pdf_text):
    prompt = (
        "请根据下面的论文内容，生成一份结构清晰、条理分明的思维导图，"
        "要求使用 Markdown 格式，适配 markmap 工具，内容突出论文结构与主要观点：\n\n"
        f"{pdf_text[:30000]}"
    )
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的学术思维导图制作专家，擅长将论文内容转化为结构清晰的思维导图。"},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        lines = content.split('\n')
        return '\n'.join(lines[1:]) if len(lines) > 1 else content
    except Exception as e:
        st.error(f"思维导图生成失败：{str(e)}")
        return "# 思维导图生成失败\n- 请检查API调用或输入内容"

# 实时对话模块（使用 DeepSeek）
def init_chat_session():
    if "result" not in st.session_state:  # 新增初始化
        st.session_state.result = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "我是论文评审助手，您可以随时提问"}]
    if "show_report" not in st.session_state:
        st.session_state.show_report = False
    if "search_results" not in st.session_state:
        st.session_state.search_results = None

def show_chat_interface(deepseek_client, paper_content):
    st.divider()
    st.subheader("💬 专家实时问询")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input("输入您的问题"):
        context = f"""
        当前论文关键内容：
        {paper_content[:2000]}
        
        历史分析结果：
        {json.dumps(st.session_state.result, ensure_ascii=False)[:1000]}
        """
        full_prompt = f"{context}\n\n专家提问：{prompt}"
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("AI正在思考..."):
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "您是论文评审专家助理，需根据分析结果专业解答"},
                        {"role": "user", "content": full_prompt}
                    ]
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"请求失败：{str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)
        st.rerun()

# 可视化模块
def generate_radar_chart(scores):
    categories = list(scores.keys())
    values = [float(scores[k]["value"]) for k in categories]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            line=dict(color="#1E90FF"),
            name="论文评分"
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=10)), angularaxis=dict(rotation=90)),
        showlegend=False,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(family="Microsoft YaHei")
    )
    return fig

# 主程序
def main():
    st.set_page_config(page_title="AI论文评审系统", page_icon="📑", layout="wide")
    st.title("📑 混合模型论文智能评审系统")
    deepseek_client = init_deepseek_client()
    local_client = init_local_model_client()
    init_chat_session()
    
    if "keywords" not in st.session_state:
        st.session_state["keywords"] = []
    # 初始化 session_state 键
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'keywords' not in st.session_state:
        st.session_state.keywords = None
    if 'mindmap' not in st.session_state:
        st.session_state.mindmap = None
    if 'report' not in st.session_state:
        st.session_state.report = None

    uploaded_file = st.file_uploader("上传学位论文PDF", type=["pdf"], help="建议上传不超过50页的PDF文件")
    
    if uploaded_file and local_client and deepseek_client:
        if "text_content" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
            with st.spinner("正在解析论文内容..."):
                st.session_state.text_content = extract_text_from_pdf(uploaded_file)
                st.session_state.uploaded_file = uploaded_file
                st.session_state.show_report = False
                st.session_state.search_results = None

        with st.sidebar:
            st.header("🔍 相关论文搜索")
            # 安全生成 search_query
            keywords = st.session_state.get("keywords", [])
            if not isinstance(keywords, (list, tuple)):
                keywords = []
            search_query = " ".join(keywords)
            if st.button("搜索相关论文", use_container_width=True):
                with st.spinner("正在搜索相关论文..."):
                    papers = search_related_papers(deepseek_client, search_query)
                    st.session_state.search_results = papers
                    st.success(f"找到{len(papers)}篇相关论文")

            if st.session_state.get("search_results"):
                st.subheader("搜索结果")
                for i, paper in enumerate(st.session_state.search_results):
                    with st.expander(f"{i+1}. {paper.get('title', '无标题')}"):
                        st.markdown(f"**作者**: {paper.get('authors', '未知')}")
                        st.markdown(f"**年份**: {paper.get('year', '未知')}")
                        st.markdown(f"**摘要**: {paper.get('summary', '无摘要')}")
                        st.markdown(f"**相关度**: {paper.get('relevance', '高')}")

        # 生成论文摘要按钮
        if st.button("🔍 生成论文摘要", use_container_width=True, disabled=not st.session_state.text_content):
            with st.spinner("正在生成摘要..."):
                summary = generate_summary_via_deepseek(deepseek_client, st.session_state.text_content, mode="青年科学基金")
                st.session_state.summary = summary

        # 显示摘要（如果已生成）
        if st.session_state.summary:
            with st.expander("查看摘要", expanded=True):
                st.success("摘要生成完毕")
                st.markdown(f"**摘要：**\n{st.session_state.summary}")

        # 提取关键词按钮
        if st.button("🔍 提取关键词", use_container_width=True, disabled=not st.session_state.text_content):
            with st.spinner("正在提取关键词..."):
                keywords = extract_keywords_via_deepseek(deepseek_client, st.session_state.text_content)
                st.session_state.keywords = keywords

        # 显示关键词（如果已提取）
        if st.session_state.keywords:
            with st.expander("查看提取的关键词", expanded=True):
                st.success("关键词提取完毕")
                st.markdown(f"**关键词：**\n{', '.join(st.session_state.keywords)}")

        # 生成论文思维导图按钮
        if st.button("🧠 生成论文思维导图", use_container_width=True, disabled=not st.session_state.text_content):
            with st.spinner("正在生成思维导图..."):
                markmap_md = ask_deepseek_for_markmap(deepseek_client, st.session_state.text_content)
                output_dir = "./markmap_output"
                file_name = "markmap (25)"
                os.makedirs(output_dir, exist_ok=True)
                markmap_auto_export(markmap_md, output_dir, file_name)
                html_path = os.path.join(output_dir, file_name + ".html")
                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.session_state.mindmap = html_content
                else:
                    st.error("思维导图生成失败，请检查 markmap_auto_export 函数或文件路径。")

        # 显示思维导图（如果已生成）
        if st.session_state.mindmap:
            with st.expander("查看思维导图", expanded=True):
                st.success("思维导图已生成！")
                st.components.v1.html(st.session_state.mindmap, height=600, scrolling=True)
                # 可选：提供下载按钮
                with open("./markmap_output/markmap.html", "rb") as f:
                    st.download_button(
                        label="下载思维导图HTML文件",
                        data=f,
                        file_name="markmap (24).html",
                        mime="text/html"
                    )

        # 生成报告解析按钮
        if st.button("🔍 生成报告解析", use_container_width=True, type="primary", disabled=not st.session_state.text_content):
            with st.spinner("正在进行深度分析..."):
                result = generate_detailed_rating_via_deepseek_and_local_model(deepseek_client, local_client, st.session_state.text_content)
                if result:
                    st.session_state.result = result
                    st.session_state.show_report = True
                else:
                    st.error("报告生成失败，请重试")

        # 显示报告解析（如果已生成）
        if st.session_state.get('show_report', False):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("## 📊 五维能力雷达图")
                fig = generate_radar_chart(st.session_state.result["scores"])
                st.plotly_chart(fig, use_container_width=True)
                pdf_path = create_pdf_report(st.session_state.result)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 下载完整报告(PDF)",
                        data=f,
                        file_name="论文评审报告.pdf",
                        mime="application/pdf"
                    )
            with col2:
                st.markdown("## 📝 增强版分析报告")
                with st.expander("📋 查看各维度评分标准", expanded=False):
                    for dim, criteria in DIMENSION_CRITERIA.items():
                        st.markdown(f"### {dim}")
                        st.markdown(f"**定义**: {criteria['description']}")
                        st.markdown("**评分标准**:")
                        for score, desc in criteria['criteria'].items():
                            st.markdown(f"- {score}: {desc}")
                for dim in DIMENSION_CRITERIA.keys():
                    with st.expander(f"{dim}分析", expanded=False):
                        st.markdown(f"**评分**: {st.session_state.result['scores'][dim]['value']}/5.0")
                        st.markdown(f"**评分依据**: {st.session_state.result['scores'][dim]['reason']}")
                        st.markdown(f"### 优点\n{st.session_state.result['comments'][dim]['strengths']}")
                        st.markdown(f"### 不足\n{st.session_state.result['comments'][dim]['weaknesses']}")
                        st.markdown(f"### 建议\n{st.session_state.result['comments'][dim]['suggestions']}")
            show_chat_interface(deepseek_client, st.session_state.text_content)

if __name__ == "__main__":
    main()
