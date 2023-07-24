import torch
import os
try:
  from llama_cpp import Llama
except:
  if torch.cuda.is_available():
      print("CUDA is available on this system.")
      os.system('CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose')
  else:
      print("CUDA is not available on this system.")
      os.system('pip install llama-cpp-python')

import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain.document_loaders import (
    EverNoteLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)
import param
from conversadocs.bones import DocChat

dc = DocChat()

##### GRADIO CONFIG ####

css="""
#col-container {max-width: 1500px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 1500px;">
    <h2>Chat with Documents 📚 - Falcon, Llama-2 and OpenAI</h2>
    <p style="text-align: center;">Upload txt, pdf, doc, docx, enex, epub, html, md, odt, ptt and pttx. 
    Wait for the Status to show Loaded documents, start typing your questions. Oficial Repository <a href="https://github.com/R3gm/ConversaDocs">ConversaDocs</a>.<br /></p>
</div>
"""

description = """
# Application Information

- Notebook for run ConversaDocs in Colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/ConversaDocs/blob/main/ConversaDocs_Colab.ipynb)

- Oficial Repository [![a](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/R3gm/ConversaDocs/)

- You can upload multiple documents at once to a single database.

- Every time a new database is created, the previous one is deleted.

- For maximum privacy, you can click "Load LLAMA GGML Model" to use a Llama 2 model. By default, the model llama-2_7B-Chat is loaded.

- This application works on both CPU and GPU. For fast inference with GGML models, use the GPU.

- For more information about what GGML models are, you can visit this notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/InsightSolver-Colab/blob/main/LLM_Inference_with_llama_cpp_python__Llama_2_13b_chat.ipynb)
"""

theme='aliabid94/new-theme'

def flag():
  return "PROCESSING..."

def upload_file(files, max_docs):
    file_paths = [file.name for file in files]
    return dc.call_load_db(file_paths, max_docs)

def predict(message, chat_history, max_k, check_memory):
        print(message)
        bot_message = dc.convchain(message, max_k, check_memory)
        print(bot_message)
        return "", dc.get_chats()

def convert():
  docs = dc.get_sources()
  data_docs = ""
  for i in range(0,len(docs),2):
    txt = docs[i][1].replace("\n","<br>")
    sc = "Archive: " + docs[i+1][1]["source"]
    try:
      pg = "Page: " + str(docs[i+1][1]["page"])
    except:
      pg = "Document Data"
    data_docs += f"<hr><h3 style='color:red;'>{pg}</h2><p>{txt}</p><p>{sc}</p>"
  return data_docs

def clear_api_key(api_key):
  return 'api_key...', dc.openai_model(api_key)


# Max values in generation
DOC_DB_LIMIT = 5
MAX_NEW_TOKENS = 2048

# Limit in HF, no need to set it
if "SET_LIMIT" == os.getenv("DEMO"):
    DOC_DB_LIMIT = 4
    MAX_NEW_TOKENS = 32

with gr.Blocks(theme=theme, css=css) as demo:
  with gr.Tab("Chat"):

    with gr.Column():
        gr.HTML(title)
        upload_button = gr.UploadButton("Click to Upload Files", file_count="multiple")
        file_output = gr.HTML()

        chatbot = gr.Chatbot([], elem_id="chatbot") #.style(height=300)
        msg = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        with gr.Row():
            check_memory = gr.inputs.Checkbox(label="Remember previous messages")
            clear_button = gr.Button("CLEAR CHAT HISTORY", )
            max_docs = gr.inputs.Slider(1, DOC_DB_LIMIT, default=3, label="Maximum querys to the DB.", step=1)

    with gr.Column():
        link_output = gr.HTML("")
        sou = gr.HTML("")

    clear_button.click(flag,[],[link_output]).then(dc.clr_history,[], [link_output]).then(lambda: None, None, chatbot, queue=False)
    upload_button.upload(flag,[],[file_output]).then(upload_file, [upload_button, max_docs], file_output).then(dc.clr_history,[], [link_output])
    
  with gr.Tab("Config llama-2 model"):
    gr.HTML("<h3>Only models from the GGML library are accepted. To apply the new configurations, please reload the model.</h3>")
    repo_ = gr.Textbox(label="Repository" ,value="TheBloke/Llama-2-7B-Chat-GGML")
    file_ = gr.Textbox(label="File name" ,value="llama-2-7b-chat.ggmlv3.q5_1.bin")
    max_tokens = gr.inputs.Slider(1, 2048, default=256, label="Max new tokens", step=1)
    temperature = gr.inputs.Slider(0.1, 1., default=0.2, label="Temperature", step=0.1)
    top_k = gr.inputs.Slider(0.01, 1., default=0.95, label="Top K", step=0.01)
    top_p = gr.inputs.Slider(0, 100, default=50, label="Top P", step=1)
    repeat_penalty = gr.inputs.Slider(0.1, 100., default=1.2, label="Repeat penalty", step=0.1)
    change_model_button = gr.Button("Load Llama GGML Model")
    
    model_verify_ggml = gr.HTML("Loaded model Llama-2")

  with gr.Tab("API Models"):

    default_model = gr.HTML("<hr>Falcon Model</h2>")
    hf_key = gr.Textbox(label="HF TOKEN", value="token...")
    falcon_button = gr.Button("Load FALCON 7B-Instruct")

    openai_gpt_model = gr.HTML("<hr>OpenAI Model gpt-3.5-turbo</h2>")
    api_key = gr.Textbox(label="API KEY", value="api_key...")
    openai_button = gr.Button("Load gpt-3.5-turbo")

    line_ = gr.HTML("<hr> </h2>")
    model_verify = gr.HTML(" ")

  with gr.Tab("About"):
    description_md = gr.Markdown(description)

  msg.submit(predict,[msg, chatbot, max_docs, check_memory],[msg, chatbot]).then(convert,[],[sou])

  change_model_button.click(dc.change_llm,[repo_, file_, max_tokens, temperature, top_p, top_k, repeat_penalty, max_docs],[model_verify_ggml])

  falcon_button.click(dc.default_falcon_model, [hf_key], [model_verify])
  openai_button.click(clear_api_key, [api_key], [api_key, model_verify])
    
demo.launch(share=True,  enable_queue=True)
