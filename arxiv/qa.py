#!/usr/bin/env python

# pip install langchain_ollama langchain_core langchain_community arxiv

from langchain_ollama.chat_models import ChatOllama
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
# Use if building own template (rather than using hub example)
#from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os


os.environ["LLAMAFILE_SERVER_BASE_URL"] = "http://localhost:11434"

tools = load_tools(
    ["arxiv"],
)

llm = ChatOllama(model="llama3.1", tools=tools)

question = "What papers on quantum computing have been published since September 2024"

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, prompt=prompt,tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=False)

# Currently the search fails, since it reverts to using internal llm data and 2024 is too late
# some prompt/template work needed to fix ...
#result=agent_executor.invoke({"input": question})

# Known working prompt - note this is a current article
result=agent_executor.invoke(
    {
        "input": "What's the paper 2501.07744 about?",
    }
)
print(result["output"])