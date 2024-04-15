from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from gradio import Blocks
import os, logging, json

from operator import itemgetter

from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        )
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableSerializable

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader

from typing import List

from workshop_utils import AICoreHandling, get_hana_connection, get_llm_model

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","all-MiniLM-L12-v2") 
DEFAULT_EF = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
TABLE_NAME_FOR_DOCUMENTS  = "CML2024WS"

BLOCK_CSS = """
gradio-app > .gradio-container {
    max-width: 100% !important;
    
}
.contain { display: flex !important; flex-direction: column !important; }
#chat_window { height: 70vh !important; }
#column_left { height: 88vh !important; }
#sql_col_left1 { height: 82vh !important;}
#arch_gallery { height: 88vh !important;}
#buttons button {
    min-width: min(120px,100%);
}
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

def call_llm(state: dict, history: list)->any:
    """ Handle LLM request and response """
    return history

def retrieve_data(vector_db: HanaDB, llm: BaseLanguageModel)->RunnableSerializable:
    """ Retrieves data from store and passes back result """
    return

def uploaded_files(state: dict, files: any)->None:
    """ Handles the uploaded pdf files and care for embedding into HANA VS """
    return
    
def clear_data(state: dict)->list:
    """ Clears the history of the chat """
    state_new = {
        "ai_core": state.get("ai_core"), "conn_data": state.get("conn_data", None), 
    }
    return [None, state_new]

def build_chat_view(conn_data: dict, ai_core: AICoreHandling)->Blocks:
    """ Build the view with Gradio blocks """
    with gr.Blocks(
            title="CML Workshop - Retrieve-Augment-Generate in a BTP-contained setting", 
            theme=gr.themes.Soft(),
            css=BLOCK_CSS
        ) as chat_view:
        state = gr.State({})
        state.value["conn_data"] = conn_data
        state.value["ai_core"] = ai_core
        with gr.Row(elem_id="overall_row") as main_screen:
            with gr.Column(scale=10, elem_id="column_left"):
                chatbot = gr.Chatbot(
                    label="Document Chat - HANA Vector Store and AI Core LLM",
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
                clear = gr.Button(value="Clear history")
        msg_box.submit(user, 
                       inputs=[state, msg_box, chatbot], 
                       outputs=[msg_box, chatbot], 
                       queue=True).then(
                            call_llm, 
                            inputs=[state, chatbot], 
                            outputs=[chatbot]
        )
        clear.click(clear_data, 
                    inputs=[state], 
                    outputs=[chatbot, state], 
                    queue=True)
        files.change(uploaded_files, 
                     inputs=[state, files],
                     outputs=[])
    return chat_view    


def main()->None:
    """ Main program of the tutorial for CML """
    args = {}
    args["host"] = os.environ.get("HOSTNAME","0.0.0.0")
    args["port"] = os.environ.get("HOSTPORT",51040)
    log_level = int(os.environ.get("APPLOGLEVEL", logging.ERROR))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level,)
        
    hana_cloud = {
        "host": os.getenv("HOST"),
        "user": os.getenv("USERNAME",""),
        "password": os.getenv("PASSWORD","") 
    }
    
    # Get ready to connect to AI Core
    ai_core = AICoreHandling()
    
    # Create chat UI
    chat_view = build_chat_view(conn_data=hana_cloud, ai_core=ai_core)
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