#!/usr/bin/env python

# pip install langchain_ollama langchain_core langchain_community arxiv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
import os
from typing import List

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor


os.environ["LLAMAFILE_SERVER_BASE_URL"] = "http://localhost:11434"

tools = load_tools(
    ["arxiv"],
)

llm = ChatOllama(model="llama3.1", tools=tools)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant"),
        ("human", f"{input}"),
        ("placeholder", "{agent_scratchpad}")
        ])
question = "Give me a list of up to 10 preprint research papers from arxiv on quantum computing published in the last 3 months."
prompt = hub.pull("hwchase17/react")
#agent = create_tool_calling_agent(llm, tools, prompt)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=False)

output=agent_executor.invoke({"input": question})

print(output)