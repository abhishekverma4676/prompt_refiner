
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import dotenv_values
import os 


class base_state(TypedDict):
    query:str
    gr_rem:str
    final_prompt:str


config = dotenv_values("Backend/Agents/.env")
api_key = config.get("GROQ_API_KEY")
def get_llm():
    llm= ChatGroq(api_key=api_key,temperature=0.5,model="openai/gpt-oss-120b")
    return llm


async def grammer(state:base_state):
    "remove grammer mistake from query"
    prompt = PromptTemplate(
        input_variables=["query"],
        template=""" 
        clean the following prompt
      remove the grammer_mistake and make it clear

query:
{query}

"""
    )
    llm =  get_llm()
    chain=  prompt | llm | StrOutputParser()

    result =   await chain.ainvoke({
        "query":state["query"]
    })
    return {
        "gr_rem":result
    }



async def prompt_enh(state:base_state):
    "enhance the  prompt "
    prompt = PromptTemplate(
        input_variables=["gr_rem"],
        template=""" 
You are a Senior Prompt Engineering Specialist with deep expertise in LLM behavior control, instruction hierarchy, and production prompt design.

Your sole task is to REWRITE the given input into optimized prompts.
You are STRICTLY FORBIDDEN from answering, explaining, or executing the input topic.

Input:
{gr_rem}

CRITICAL RULES (NON-NEGOTIABLE):

- DO NOT answer, explain, define, summarize, or provide facts about the topic.
- DO NOT add new information beyond restructuring the intent.
- If the input is a question, convert it into a prompt that asks another LLM to answer it.
- Your output must ONLY contain rewritten prompts, never task results.
- Any factual content about the topic itself makes the response INVALID.

Objective:

Transform the input into three distinct, high-quality, production-ready prompts.
Each prompt must preserve the original intent while improving clarity, structure, and effectiveness.

Deliverables (ALL REQUIRED):

1. Detailed Prompt
   - Written in multiple clear paragraphs
   - Explicitly define:
     • Role of the LLM
     • Objective
     • Context
     • Constraints
     • Assumptions
     • Expected output format
   - Suitable for high-stakes or long-form generation

2. Concise Prompt
   - Short and highly focused
   - Retains only the core objective and key constraints
   - Optimized for quick execution with minimal tokens

3. Instructional Prompt
   - Written as numbered, step-by-step instructions
   - Clearly guides the model’s process and output structure
   - Emphasizes reasoning discipline and boundaries

Output Format (STRICT — follow exactly):

Detailed Prompt:
<rewritten detailed prompt>

Concise Prompt:
<rewritten concise prompt>

Instructional Prompt:
<rewritten instructional prompt>


"""
    )
    llm =  get_llm()
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke({
        "gr_rem":state["gr_rem"]
    })
    return {
        "final_prompt":result
    }


graph= StateGraph(base_state)

graph.add_node("mistake_remover",grammer)
graph.add_node("prompt_enhancer",prompt_enh)


graph.add_edge(START,"mistake_remover")
graph.add_edge("mistake_remover","prompt_enhancer")
graph.add_edge("prompt_enhancer",END)

agent = graph.compile()



async def refiner(query:str):
    intial_state={
        "query":query
    }

    async for message_chunk, metadata in agent.astream(intial_state, stream_mode="messages"):
        if message_chunk.content:
            yield message_chunk.content



