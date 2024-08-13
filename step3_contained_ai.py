from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from gradio import Blocks
import os, logging, json

from operator import itemgetter

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableSerializable, RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate

from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders.pdf import PyMuPDFLoader

from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.proxy.langchain import init_llm, init_embedding_model

from typing import List

from workshop_utils import get_hana_connection

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","text-embedding-3-small") 
TABLE_NAME_FOR_DOCUMENTS  = "PWC2024"

RAG_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as detailed as necessary. Write the sources (document and page or paragraph if available) at the bottom as footnotes. Use markdown to structure the output.
Question: {input}
Context: {context}
Answer:"""

LOGO_MARKDOWN = f"""
### PwC SAP Workshop August 2024
![PwC SAP Workshop](file/img/cml2024.webp)
"""

BLOCK_CSS = """
gradio-app > .gradio-container {
    max-width: 100% !important;
}
.contain { display: flex !important; flex-direction: column !important; }
#chat_window { height: calc(100vh - 112px - 64px) !important; }
#genai_tab { flex-grow: 1 !important; overflow: auto !important; }
#column_left { height: 100vh !important; }
#arch_gallery { height: 88vh !important;}
#login_count_df { height: 88vh !important;}
footer {
    display:none !important
}
"""


def user(state: dict, user_message: str, history: list)->tuple:
    """ Handle user interaction in the chat window """
    state["skip_llm"] = False
    if len(user_message) <= 0:
        state["skip_llm"] = True
        return "", history, None
    rv =  "", history + [[user_message, None]]
    return rv


def call_llm(state: dict, history: list, rag_active: bool)->any:
    """ Handle LLM request and response """
    do_stream = True
    my_chain = None
    if state["skip_llm"] == True:
        yield history
        return history
    history[-1][1] = ""
    if not state.get("memory", None):
        state["memory"]=ConversationBufferMemory(memory_key="history", return_messages=True)
        state["memory"].clear()
    if rag_active:
        # Below part is just temporary.----------------
        pass
        # ------------------ End of temporary part
    else:
        my_prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content="You are an assistant speaking users in the age group of 15 to 20."),
                        MessagesPlaceholder(variable_name="history"),
                        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}"))
                    ],
                )    
        my_chain = (
            {
                "history": RunnableLambda(state["memory"].load_memory_variables) | itemgetter("history"), 
                "input": RunnablePassthrough()
            }
            | my_prompt
            | LLM
            | StrOutputParser()
        )
    query = history[-1][0]
    try:
        if my_chain:
            if do_stream:
                response: str
                input_value = query if rag_active else {"input": query}
                for response in my_chain.stream(input_value): 
                    history[-1][1] += response
                    logging.debug(history[-1][1])
                    yield history
            else:
                response=my_chain.invoke({"query": query}) 
                history[-1][1] += response
                logging.debug(history[-1][1])
                yield history
        else: # Chain not active
            history[-1][1] = "This function is not active yet."
            yield history
    except Exception as e:
        history[-1][1] += str(f"😱 Oh no! It seems the LLM has some issues. Error was: {e}.")
        yield history
    finally:        
        state["memory"].save_context({"input": history[-1][0]},{"output": history[-1][1]})
        return history 


def retrieve_data(vector_db: HanaDB, llm: BaseLanguageModel)->RunnableSerializable:
    """ Retrieves data from store and passes back result """
    pass


def format_docs(docs: List[Document]) -> str:
    """ Concatenates the result documents after vector db retrieval """
    result = ""
    for doc in docs:
        source = f"Source document '{doc.metadata['source']}'" if 'source' in doc.metadata else ""
        page = f"Page {doc.metadata['page']}" if 'page' in doc.metadata else ""
        result +=  f"Below text is from {source} {page}:\n-------------------------------------------\n{doc.page_content}\n"
    return result


def uploaded_files(state: dict, files: List[str])->None:
    """ Handles the uploaded pdf files and care for embedding into HANA VS """
    success = False
    documents = []
    if not files: # Happens when the list is cleared
        return [gr.Checkbox(visible=False, value=False)]
    for file in files:
        if file.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path=file, extract_images=False) 
        else:
            raise ValueError('File format not supported. Please provide a .pdf file.')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=100, 
            length_function=len, 
            is_separator_regex=False
        )
        docs=loader.load_and_split(text_splitter)
        for doc in docs:
            if doc.metadata.get("source", None) != None:
                doc.metadata["source"]=os.path.basename(file)
            if doc.metadata.get("page", None) != None:
                doc.metadata["page"]=int(doc.metadata["page"]) + 1
        documents.extend(docs)
    msg = f"Uploaded {len(files)} file(s). Split into {len(documents)} documents."
    logging.info(msg)
    gr.Info(msg)
    
    if not state.get("connection"):
        state["connection"] = get_hana_connection(conn_params=state["conn_data"])
    vector_db = HanaDB(
        connection=state["connection"],
        embedding=EMBMOD, 
        table_name=TABLE_NAME_FOR_DOCUMENTS,
        vector_column_length=0,  
    )
    try:
        vector_db.delete(filter={})
    except Exception as e:
        logging.warning(f"Deleting embedding entries failed with error {e}. Maybe there were no embeddings?")
    # Add the documents which we uploaded and split
    try:
        vector_db.add_documents(documents=documents)
        msg=f"Embedded {len(documents)} documents in table {TABLE_NAME_FOR_DOCUMENTS}."
        logging.info(msg)
        gr.Info(msg)
        success = True
    except Exception as e:
        logging.error(f"Adding document embeddings failed with error {e}.")
    finally:
        return [gr.Checkbox(visible=success, value=success)]
    
    
def clear_data(state: dict)->list:
    """ Clears the history of the chat """
    state_new = {
        "conn_data": state.get("conn_data", None), 
    }
    return [None, state_new]


def build_chat_view(conn_data: dict)->Blocks:
    """ Build the view with Gradio blocks """
    with gr.Blocks(
            title="PwC Workshop - Learning SAP Generative AI Hub and HANA VS", 
            theme=gr.themes.Soft(),
            css=BLOCK_CSS
        ) as chat_view:
        state = gr.State({})
        state.value["conn_data"] = conn_data
        with gr.Row(elem_id="overall_row") as main_screen:
            with gr.Column(scale=10, elem_id="column_left"):
                chatbot = gr.Chatbot(
                    label="Document Chat - HANA Vector Store and GenAI LLM",
                    elem_id="chat_window",
                    bubble_full_width=False,
                    show_copy_button=True,
                    show_share_button=True,
                    avatar_images=(None, "./img/saplogo.png")
                )
                with gr.Row(elem_id="input_row") as query_row:
                    msg_box = gr.Textbox(
                        scale=9,
                        elem_id="msg_box",
                        show_label=False,
                        max_lines=5,
                        placeholder="Enter text and press ENTER",
                        container=False,
                        autofocus=True                    )
            with gr.Column(scale=3, elem_id="column_right") as column_right:
                files = gr.File(label="RAG File Upload", file_count="multiple", file_types=[".pdf"])
                rag_chkbox = gr.Checkbox(label="RAG active", value=False, visible=True, elem_id="rag_chkbox")
                clear = gr.Button(value="Clear history")
                cml2024img = gr.Markdown(value=LOGO_MARKDOWN, elem_id="cml2024_box")
        msg_box.submit(user, 
                       inputs=[state, msg_box, chatbot], 
                       outputs=[msg_box, chatbot], 
                       queue=True).then(
                            call_llm, 
                            inputs=[state, chatbot, rag_chkbox], 
                            outputs=[chatbot]
        )
        clear.click(clear_data, 
                    inputs=[state], 
                    outputs=[chatbot, state], 
                    queue=True)
        files.change(uploaded_files, 
                     inputs=[state, files],
                     outputs=[rag_chkbox])
    return chat_view    


def main()->None:
    """ Main program of the tutorial for workshop """
    args = {}
    args["host"] = os.environ.get("HOSTNAME","0.0.0.0")
    args["port"] = os.environ.get("HOSTPORT",51040)
    log_level = int(os.environ.get("APPLOGLEVEL", logging.ERROR))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level,)
        
    hana_cloud = {
        "host": os.getenv("HANA_DB_ADDRESS"),
        "user": os.getenv("HANA_DB_USER",""),
        "password": os.getenv("HANA_DB_PASSWORD","") 
    }
    
    # Connect to GenAI Hub
    genai_proxy = GenAIHubProxyClient()
    global LLM, EMBMOD
    LLM = init_llm(
        model_name="gpt-4o", 
        proxy_client=genai_proxy, 
        temperature=0.5, 
        top_p=0.7, 
        max_tokens=500
    )
    EMBMOD = init_embedding_model(
        model_name=EMBEDDING_MODEL, 
        proxy_client=genai_proxy
    )
    
    # Create chat UI
    chat_view = build_chat_view(conn_data=hana_cloud)
    # Queue input of each user
    chat_view.queue(max_size=10)
    # Start the Gradio server
    chat_view.launch(
        debug=False,
        show_api=False,
        server_name=args["host"],
        server_port=args["port"],
        allowed_paths=["./img"]
    )
    
if __name__ == "__main__":
    main()