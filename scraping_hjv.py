import os
import shutil
import warnings
import logging
# from dotenv import find_dotenv, load_dotenv # Puedes comentar esta línea si ya no la usas en este archivo
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import asyncio
from langchain_community.document_loaders import PyPDFLoader
#Configuración inicial de logs y warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.ERROR)
from dotenv import find_dotenv, load_dotenv
import os
import asyncio
import warnings
import logging
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_hyperbrowser import HyperbrowserScrapeTool
import json
import pandas as pd

load_dotenv(find_dotenv())

   
google_api_key = "x" 
os.environ["GOOGLE_API_KEY"] = google_api_key
 



if not google_api_key: 
    raise ValueError("La variable GOOGLE_API_KEY está vacía después de la asignación directa.")




try:
        llm_gemini_instance = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        embeddings_model_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Modelos de Langchain (LLM y Embeddings) inicializados en rag_system.")
        

except Exception as e:
        print(f"ERROR: Falló la inicialización de ChatGoogleGenerativeAI o GoogleGenerativeAIEmbeddings. Mensaje: {e}")
        print("Asegúrate de que los nombres de los modelos sean correctos y que la API Key tenga acceso a ellos.")

try:
        Hyperbrowser_api_key = "x"
        os.environ["HYPERBROWSER_API_KEY"] = Hyperbrowser_api_key
        print("Hyperbrowser configurado en rag_system.")

except Exception as e:
        print(f"ERROR: Falló la inicialización de Hyperbrowser. Mensaje: {e}")
        print("Asegúrate de que la API Key de Hyperbrowser sea válida.")


from langchain_hyperbrowser import HyperbrowserCrawlTool
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


crawl_tool = HyperbrowserScrapeTool()
page_content = crawl_tool.invoke({
    
    "url": "https://www.bvc.com.co/mercado-local-en-linea?tab=renta-variable_mercado-local",
    "render": True,  
    "wait_time": 20   

    })

# Paso 2: Crear una cadena para extraer el dato
extraction_prompt = PromptTemplate.from_template(
    """Eres un experto en extracción de datos de tablas. Tu tarea es extraer la información organizarla en formato JSON.

Para cada fila de la tabla, debes crear un objeto JSON con las siguientes claves:

- "nemotecnico"
- "ultimo_precio"
- "variacion_porcentual"
- "volumenes"
- "cantidad"
- "variacion_absoluta" 
- "precio_apertura"
- "precio_maximo"
- "precio_minimo"

Asegúrate de que los valores numéricos se mantengan como números y los textos como cadenas de texto. No incluyas información adicional, solo el JSON completo.
 """
    "Texto: {page_content}"
)



llm=llm_gemini_instance
extraction_chain = extraction_prompt | llm


extracted_info = extraction_chain.invoke({"page_content": page_content})

print(extracted_info)


try:
    data_list=json.loads(extracted_info)

    df=pd.DataFrame(data_list)

    print("\n--- DataFrame de Pandas Creado con Éxito ---")
    print(df.head())
except Exception as e:
    print(f"ERROR: Falló la conversión a lista JSON. Mensaje: {e}")
    print("Asegúrate de que el texto extraído sea un JSON válido.")
    print("Salida del Agente:")


# Exportar la lista de datos a un archivo JSON



