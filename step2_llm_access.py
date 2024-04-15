from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_community.llms.vllm import VLLMOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from operator import itemgetter
from workshop_utils import AICoreHandling, get_llm_model

SYS_TEMPLATE = """GPT4 Correct System: You're an assistant explaining everything in a funny way.<|end_of_turn|>"""
HUMAN_TEMPLATE = """GPT4 Correct User: The question to you is: {query}. 
    Provide a 1 sentence answer.<|end_of_turn|>
    GPT4 Correct Assistant:
"""

def get_llm_response(query: str, llm: any)->str:
    """ Getting a reply on a question from the llm """
    query_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(SYS_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)  
        ],
        input_variables=['query'],
    )
    test_chain=(
            {
                "query": itemgetter("query")
            } 
            | query_prompt 
            | llm
        )    
    response=test_chain.invoke({"query": query})
    return response

def main()->None:
    """ A simple program to test an LLM running on AI Core """
    ai_core = AICoreHandling()
    llm = get_llm_model(ai_core=ai_core, temperature=0.5)
    
    question = "What is SAP's business?"
    
    answer_from_llm=get_llm_response(query=question, llm=llm)
    print("------ * * * >>>>>>")
    print(f"Question: {question}.\n\nAnswer: {answer_from_llm}")
    print("<<<<<< * * * ------")

if __name__ == "__main__":
    main()