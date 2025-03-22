"""This file was generated using `langgraph-gen` version 0.0.3.

This file provides a placeholder implementation for the corresponding stub.

Replace the placeholder implementation with your own logic.
"""

from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

from stub import CustomAgent

# from langchain_experimental.tools import PythonREPLTool
# from langchain.agents import AgentExecutor
# from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
# from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd
load_dotenv()


class Search(BaseModel):
    search_decision: bool = Field(
        None, description="search decision"
    )
    search_query: str | None = Field(None, description="search query")

class SomeState(TypedDict):
    # define your attributes here
    user_msg: str
    search_response: str
    instruction_for_coding_llm: str
    generated_code: str
    search_decision: bool
    search_query: str
    search_result: list
    df: pd.DataFrame


sys_prompt_base_llm = """
You are an AI model that generates a prompt for coding model to create a DataFrame with learning path for area or subject provided from user and result found on the internet

### Instructions:
1. Take the user’s input for the area or subject they want to learn about.
2. You also can use a tool to search the internet for additional information.
3. instruct the coding model to generate a pandas DataFrame with the learning path and columns that are necessary to include for user based on data you found.
4. Do not include code for dataframe generation, only instruction for coding model
"""
system_base_message = SystemMessagePromptTemplate.from_template(sys_prompt_base_llm)
human_base_message = HumanMessagePromptTemplate.from_template("""### Input from user 
                                                            {user_msg}
                                                            {search_result}
                                                        """)
prompt_base_llm = ChatPromptTemplate.from_messages([system_base_message, human_base_message])

sys_prompt_coding_llm = """
You are an AI coding model that generates code to produce pandas DataFrame given instructions of learning path recommendation
return only code without any special markups
"""
sys_code_msg = SystemMessagePromptTemplate.from_template(sys_prompt_coding_llm)
human_code_msg = HumanMessagePromptTemplate.from_template("""### Instructions
                                                            {instruction_for_coding_llm}
                                                            call dataframe 'df'
                                                        """)
prompt_code_llm = ChatPromptTemplate.from_messages([sys_code_msg, human_code_msg])


sys_cond_msg = SystemMessagePromptTemplate.from_template("""
Choose whether we need to use websearch or not to refine our instructions for coding llm 
and return search_query if websearch needed
""")
human_cond_msg = HumanMessagePromptTemplate.from_template("""
                                                            ### user input
                                                            {user_msg}
                                                          """)
prompt_cond_llm = ChatPromptTemplate.from_messages([sys_cond_msg, human_cond_msg])

# base agent to prepare prompt for coding llm to generate dataframe
base_llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
base_agent = prompt_base_llm | base_llm | StrOutputParser()

# my conditional agent to determine whether we need to search internet in particular case
cond_agent = prompt_cond_llm | base_llm.with_structured_output(Search)

# coding agent
code_llm = ChatGroq(model="qwen-2.5-coder-32b")
code_agent = prompt_code_llm | code_llm | StrOutputParser()


# Define stand-alone functions
def extractor(state: SomeState) -> dict:
    print("In node: extractor")
    instruction = base_agent.invoke({"user_msg": state["user_msg"],
                                     "search_result": state.get("search_result", [])})

    state["instruction_for_coding_llm"] = instruction

    res = cond_agent.invoke({"user_msg": state["user_msg"]})
    state.update({"search_decision": res.search_decision})
    state.update({"search_query": res.search_query})

    return state


def code_generator(state: SomeState) -> dict:
    print("In node: code generator")
    generated_code = code_agent.invoke({"instruction_for_coding_llm": state["instruction_for_coding_llm"].split("</think>")[-1]})
    # generated_code += "\nimport streamlit as st\nst.dataframe(df)"
    state["generated_code"] = generated_code

    namespace = {}
    exec(state["generated_code"], namespace)
    state["df"] = namespace["df"]
    return state


def websearch_tool(state: SomeState) -> dict:
    import pandas as pd
    print("In node: websearch_tool")
    web_search_tool = TavilySearchResults(k=1)
    documents = []

    web_results = web_search_tool.invoke({"query": state["search_query"]})
    documents.extend(
        [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]
    )

    state["search_result"] = documents
    return state


def conditional_edge_1(state: SomeState) -> str:
    print("In Node: conditional_edge_1")
    return "code generator" if (state["search_decision"] == False) or (len(state.get("search_result", [])) > 0) else "websearch_tool"



agent = CustomAgent(
    state_schema=SomeState,
    impl=[
        ("extractor", extractor),
        ("code_generator", code_generator),
        ("websearch_tool", websearch_tool),
        ("conditional_edge_1", conditional_edge_1),
    ],
)

compiled_agent = agent.compile()

# import streamlit as st
#     # tools = [PythonREPLTool()]
#     # executor = AgentExecutor(agent=compiled_agent, tools=tools, verbose=True)
# state = compiled_agent.invoke(input={
#         "user_msg": "want to learn about discrete math"
# })
# st.dataframe(state["df"])
#     # res = PythonREPLTool().run(res["generated_code"])
#     # print(res)
#     # print(df)
#     # output = executor.run(state["generated_code"])
#     # a = 0
