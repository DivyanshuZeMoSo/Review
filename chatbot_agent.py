# chatbot_agent.py

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities.python import PythonREPL
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory


# === 1. Setup LLM ===
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model_name="llama-3.1-8b-lexi-uncensored-v2",
    temperature=0.7
)

# === 2. Define Tools ===

python_repl = PythonREPL()
calculator_tool = Tool(
    name="Calculator",
    func=python_repl.run,
    description="Useful for math calculations and code evaluation."
)

search_tool = DuckDuckGoSearchRun()
search = Tool(
    name="WebSearch",
    func=search_tool.run,
    description="Useful for searching current events and general information."
)

file_reader = ReadFileTool(root_dir=".")
read_tool = Tool(
    name="ReadFile",
    func=file_reader.run,
    description="Reads the content of a local text file. Input should be the file path."
)

def pdf_qa_tool(input_str: str) -> str:
    if "|" not in input_str:
        return "[PDF_QA] Error: Input must be in format 'path/to/file.pdf|your question'"
    try:
        path, query = input_str.split("|", 1)
        path = path.strip()
        query = query.strip()

        if not path.lower().endswith(".pdf"):
            return "[PDF_QA] Only PDF files are supported."

        loader = PyPDFLoader(path)
        docs = loader.load()
        if not docs:
            return "[PDF_QA] No content found in the PDF."

        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        if not chunks:
            return "[PDF_QA] No usable chunks extracted from the PDF."

        embeddings = OpenAIEmbeddings(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio"
        )
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever(search_type="similarity", k=3)

        results = retriever.get_relevant_documents(query)
        if not results:
            return "[PDF_QA] No relevant content found in the PDF."

        chain = load_qa_chain(llm, chain_type="stuff")
        return chain.run(input_documents=results, question=query)
    except Exception as e:
        pass

pdf_tool = Tool(
    name="PDF_QA",
    func=pdf_qa_tool,
    description="Ask a question from a local PDF. Input format: 'path/to/file.pdf|your question'"
)

tools = [calculator_tool, search, read_tool, pdf_tool]

# === 3. Prompt Template ===
prompt = ChatPromptTemplate.from_messages([
    ("system",
"""You are a ReAct-style agent that uses tools to solve problems.

You MUST use this format strictly:

Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>

Once you know the answer, respond with:

Final Answer: <your final answer>

STOP immediately after Final Answer. Do not output more thoughts or actions.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")
]).partial(
    tool_names=", ".join([tool.name for tool in tools]),
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
)

# === 4. Agent Setup ===
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=1,
    early_stopping_method="generate"
)

# === 5. CLI Interface ===
print("Your local AI assistant is running! Type 'exit' to stop.\n")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    try:
        response = agent_executor.invoke({"input": user_input})
        raw_output = response.get("output", "").strip()

        # === Filter known noisy substrings ===
        filtered = raw_output
        for noisy in [
            "Invalid Format:", "Invalid or incomplete response", 
            "Parsing LLM output", "For troubleshooting", 
            "https://python.langchain.com/docs/troubleshooting/errors/",
            "For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE Invalid or incomplete response"
        ]:
            filtered = filtered.replace(noisy, "")

        # === Extract Final Answer cleanly ===
        if "Final Answer:" in filtered:
            final = filtered.split("Final Answer:", 1)[1].strip()
            print(f"Bot: {final}\n")
        else:
            print("Bot: [No final answer returned. Try rephrasing.]\n")

    except Exception as e:
        pass
