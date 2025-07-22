# chatbot_multi_agent.py
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.utilities import PythonREPL
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# === LLM Shared ===
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model_name="llama-3.1-8b-lexi-uncensored-v2",
    temperature=0.7
)

# === Tools Definitions ===

def pdf_qa_tool(input_str: str) -> str:
    if "|" not in input_str:
        return "[PDF_QA] Error: Input must be in format 'path/to/file.pdf|your question'"
    try:
        path, query = input_str.split("|", 1)
        loader = PyPDFLoader(path.strip())
        docs = loader.load()

        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        embeddings = OpenAIEmbeddings(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
        db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=".chroma_store")
        retriever = db.as_retriever(search_type="similarity", k=3)

        results = retriever.get_relevant_documents(query.strip())
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain.run(input_documents=results, question=query)
    except Exception as e:
        return f"[PDF_QA] Error processing the PDF: {e}"

@tool
def smart_search(query: str) -> str:
    """Search current info or fallback to LLM if needed."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        duckduckgo = DuckDuckGoSearchRun()
        result = duckduckgo.run(query)
        if not result.strip():
            raise ValueError("Empty search result.")
        return result
    except Exception:
        response = llm.invoke(f"You are an AI assistant. Answer this using your knowledge: {query}")
        return f"(Fallback LLM) {response.content.strip()}"

# Specialized Tools
math_tool = Tool("Calculator", PythonREPL().run, "Useful for math/code eval")
search_tool = Tool("WebSearch", smart_search, "Search current info or fallback to LLM if needed.")
file_tool = Tool("ReadFile", ReadFileTool(root_dir=".").run, "Reads local file content")
pdf_tool = Tool("PDF_QA", pdf_qa_tool, "Ask a question from PDF. Format: 'path|question'")

# === Prompt Template ===
def agent_prompt(tools):
    return ChatPromptTemplate.from_messages([
        ("system",
    """         You are a powerful, knowledgeable AI assistant capable of using external tools to solve complex problems. You follow the ReAct (Reasoning + Acting) framework to think step-by-step and act precisely.

You MUST follow this strict format:

Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>

Once you have the information needed, respond with:

Final Answer: <your complete, helpful response to the user>

RULES:
- You MUST include "Final Answer:" before your final response.
- STOP after you write "Final Answer" — do NOT include extra thoughts.
- If tools fail or are unavailable, still generate your best answer from your own knowledge.
- Be concise, clear, and specific.
- Do not make up tools that don't exist.

You have access to the following tools:
{tools}

Only use these tools. For any other task, rely on your own reasoning.
"""
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ]).partial(
        tool_names=", ".join([tool.name for tool in tools]),
        tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    )

# === Create Sub-Agents ===
def make_agent(name, tools):
    return AgentExecutor(
        agent=create_react_agent(llm=llm, tools=tools, prompt=agent_prompt(tools)),
        tools=tools,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        handle_parsing_errors=True,
        max_iterations=1,
        early_stopping_method="force",
        verbose=True
    )

math_agent = make_agent("MathAgent", [math_tool])
search_agent = make_agent("SearchAgent", [search_tool])
file_agent = make_agent("FileAgent", [file_tool])
pdf_agent = make_agent("PDFAgent", [pdf_tool])

# === Router Logic ===
def route_input(user_input):
    lower = user_input.lower()
    if any(k in lower for k in ["calculate", "math", "solve", "code"]):
        return math_agent
    elif any(k in lower for k in ["search", "news", "find", "current"]):
        return search_agent
    elif ".pdf|" in user_input:
        return pdf_agent
    elif ".txt" in lower or "read" in lower:
        return file_agent
    else:
        return search_agent

# === CLI Loop ===
print("Your multi-agent assistant is running! Type 'exit' to stop.\n")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    try:
        agent = route_input(user_input)
        result = agent.invoke({"input": user_input})
        output = result.get("output", "").strip()

    except Exception as e:
        pass