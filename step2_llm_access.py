from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.proxy.langchain import init_llm

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

import logging, os

from operator import itemgetter


SYS_TEMPLATE = """You are a funny assistant that can explain complicted things in a simple way for children."""
HUMAN_TEMPLATE = """The question you got is: {query}."""
MODEL_NAME = os.environ.get("LLM_NAME", "gpt-4o")

def get_llm_response(query: str, llm: BaseLanguageModel)->AIMessage:
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
    msg=test_chain.invoke({"query": query})
    return msg

def main()->None:
    """ A simple program to test an LLM running on AI Core """
    log_level = int(os.environ.get("APPLOGLEVEL", logging.INFO))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level,)
    
    # Connect to GenAI Hub
    genai_proxy = GenAIHubProxyClient()
    llm = init_llm(
        model_name=MODEL_NAME, 
        proxy_client=genai_proxy, 
        temperature=0.5, 
        top_p=0.7, 
        max_tokens=500
    )
        
    question = "What is SAP's business?"
    
    answer_from_llm=get_llm_response(query=question, llm=llm)
    logging.info(f"Answer: {answer_from_llm.content}\nResponse metadata: {answer_from_llm.response_metadata.get("token_usage")}")

if __name__ == "__main__":
    main()