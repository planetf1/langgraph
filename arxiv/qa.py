#!/usr/bin/env python

# pip install langchain_ollama langchain_core langchain_community arxiv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
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

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''


question = "Give me a list of up to 10 papers from arxiv on quantum computing published since September 2024"

#prompt = PromptTemplate.from_template(template)
prompt = hub.pull("hwchase17/react")
#agent = create_tool_calling_agent(llm, tools, prompt)
agent = create_react_agent(llm, prompt=prompt,tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=False)

output=agent_executor.invoke({"input": question})

print(output)