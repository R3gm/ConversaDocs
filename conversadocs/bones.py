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
import param
import os
import torch
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

#YOUR_HF_TOKEN = os.getenv("My_hf_token")
llm_api=HuggingFaceHub(
    huggingfacehub_api_token=os.getenv("My_hf_token"),
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={
        "temperature":0.2,
        "max_new_tokens":500,
        "top_k":50,
        "top_p":0.95,
        "repetition_penalty":1.2,
        },), #ChatOpenAI(model_name=llm_name, temperature=0)


#alter
def load_db(files):
    EXTENSIONS = {
        ".txt": (TextLoader, {"encoding": "utf8"}),
        ".pdf": (PyPDFLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".enex": (EverNoteLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".odt": (UnstructuredODTLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
    }



    # select extensions loader
    documents = []
    for file in files:
      ext = "." + file.rsplit(".", 1)[-1]
      if ext in EXTENSIONS:
          loader_class, loader_args = EXTENSIONS[ext]
          loader = loader_class(file, **loader_args)
          documents.extend(loader.load())
      else:
        pass

    # load documents
    if documents == []:
        loader_class, loader_args = EXTENSIONS['.txt']
        loader = loader_class('demo_docs/demo.txt', **loader_args)
        documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # define embedding
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # all-mpnet-base-v2 #embeddings = OpenAIEmbeddings()

    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    return db

def q_a(db, chain_type="stuff", k=3, llm=None):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa



class DocChat(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    llm = llm_api[0]
    k_value = param.Integer(3)


    def __init__(self,  **params):
        super(DocChat, self).__init__( **params)
        self.loaded_file = "demo_docs/demo.txt"
        self.db = load_db(self.loaded_file)
        self.qa = q_a(self.db, "stuff", self.k_value, self.llm)
        

    def call_load_db(self, path_file, k):
        if not os.path.exists(path_file[0]):  # init or no file specified
            return "No file loaded"
        else:
          try:
            self.db = load_db(path_file)
            self.loaded_file = path_file
            self.qa = q_a(self.db, "stuff", k, self.llm)
            self.k_value = k
            #self.clr_history()
            return f"New DB created and history cleared | Loaded File: {self.loaded_file}"
          except:
            return f'No valid file'


    # chat
    def convchain(self, query, k_max, recall_previous_messages):
        if k_max != self.k_value:
          print("Maximum querys changed")
          self.qa = q_a(self.db, "stuff", k_max, self.llm)
          self.k_value = k_max

        if not recall_previous_messages:
          self.clr_history()

        try:
          result = self.qa({"question": query, "chat_history": self.chat_history})
        except:
          self.default_falcon_model()
          self.qa = q_a(self.db, "stuff", k_max, self.llm)
          result = self.qa({"question": query, "chat_history": self.chat_history})
        
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        return self.answer

    def change_llm(self, repo_, file_, max_tokens=16, temperature=0.2, top_p=0.95, top_k=50, repeat_penalty=1.2, k=3):

        if torch.cuda.is_available():
            try:
              model_path = hf_hub_download(repo_id=repo_, filename=file_)

              self.llm = LlamaCpp(
                  model_path=model_path,
                  n_ctx=1000,
                  n_batch=512,
                  n_gpu_layers=35,
                  max_tokens=max_tokens,
                  verbose=False,
                  temperature=temperature,
                  top_p=top_p,
                  top_k=top_k,
                  repeat_penalty=repeat_penalty,
                  )
              self.qa = q_a(self.db, "stuff", k, self.llm)
              self.k_value = k
              return f"Loaded {file_} [GPU INFERENCE]"
            except:
              return "No valid model"
        else:
            try:
              model_path = hf_hub_download(repo_id=repo_, filename=file_)

              self.llm = LlamaCpp(
                  model_path=model_path,
                  n_ctx=1000,
                  n_batch=8,
                  max_tokens=max_tokens,
                  verbose=False,
                  temperature=temperature,
                  top_p=top_p,
                  top_k=top_k,
                  repeat_penalty=repeat_penalty,
                  )
              self.qa = q_a(self.db, "stuff", k, self.llm)
              self.k_value = k
              return f"Loaded {file_} [CPU INFERENCE SLOW]"
            except:
              return "No valid model"

    def default_falcon_model(self):
      self.llm = llm_api[0]
      self.qa = q_a(self.db, "stuff", self.k_value, self.llm)
      return "Loaded model Falcon 7B-instruct [API FAST INFERENCE]"

    def openai_model(self, API_KEY):
        self.llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name='gpt-3.5-turbo')
        self.qa = q_a(self.db, "stuff", self.k_value, self.llm)
        API_KEY = ""
        return "Loaded model OpenAI gpt-3.5-turbo [API FAST INFERENCE] | If there is no response from the API, Falcon 7B-instruct will be used."

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return print("Last question to DB: no DB accesses so far")
        return self.db_query

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return
        #rlist=[f"Result of DB lookup:"]
        rlist=[]
        for doc in self.db_response:
          for element in doc:
            rlist.append(element)
        return rlist

    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history:
            return "No History Yet"
        #rlist=[f"Current Chat History variable"]
        rlist=[]
        for exchange in self.chat_history:
            rlist.append(exchange)
        return rlist

    def clr_history(self,count=0):
        self.chat_history = []
        return "HISTORY CLEARED"
