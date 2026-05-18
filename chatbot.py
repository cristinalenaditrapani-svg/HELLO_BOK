# il mio chatbot online

# streamlit
import streamlit as st
import pdfplumber

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.markdown(
    """
    <style>
    .stApp {
        background-color: #B84B4B;
        color: #eeebe3;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.header("Hello Bok")

st.image("BOK.png", width=100)
with st.sidebar:
    st.title("il mio CV")
    documento = st.file_uploader("Carica il tuo pdf", type=["pdf"])
    
# Estrazione del contenuto e spezzettamento
if documento is not None:
    with pdfplumber.open(documento) as pdf:
        #st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
        testo = ""
        for pagina in pdf.pages:
            testo = testo + pagina.extract_text() + "\n"
            # testo += pagina.extract_text() + "\n"
    #st.write(testo)

    taglierina = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)
    
    frammenti = taglierina.split_text(testo)
    #st.write(f"Totale frammenti creati: {len(frammenti)}")
    #st.write(frammenti)
 # Generiamo gli embeddings
    # Puoi cambiare OpenAIEmbeddings e metterne altri
    # https://docs.langchain.com/oss/python/integrations/embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    st.write("Embedding recuperati!")

    # Salviamo gli embeddings in un vector store o vector db (es. FAISS, Pinecone, etc.)
    vettori = FAISS.from_texts(frammenti, embedding=embeddings)
 # Richiesta utente
    domanda_utente = st.text_input("Fai una domanda sul documento caricato:")

    def formatta_documento(documenti):
        return "\n\n".join([documento.page_content for documento in documenti])
    
    # Quando userò il prompt, qui dentro dovrà essere inserito qualcosa chiamato "context"
    # e qualcosa chiamata "question"
    # Qui è come nei roles di ChatGPT, ma qui siamo in Langchain
    # e la struttura è più semplice: "system" e "human"
    # Attenzione che nelle stringhe ''' vengono conservati spazi e indentazioni!
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         '''Sei HelloBok, un assistente AI per la selezione del personale di WeRoad.
WeRoad è un'azienda di viaggi di gruppo che organizza esperienze in tutto 
il mondo. I Tour Operator sono figure chiave: guidano i viaggi sul campo, 
gestiscono i partecipanti e rappresentano i valori del brand.

Il tuo compito è supportare il recruiter nello screening iniziale dei CV. 
Per ogni candidato recuperato:
- Scrivi una sintesi di massimo 5 righe
- Evidenzia le aree geografiche di esperienza
- Indica se ha già collaborato con WeRoad (sì/no/non specificato)
- Assegna un commento motivato sui punti di forza e sui gap rispetto 
  ai criteri indicati dal recruiter

Non decidere chi assumere. Proponi e spiega. 
Il recruiter ha sempre l'ultima parola.
Rispondi sempre in italiano.
    Se non conosci la risposta, dì semplicemente 'Non lo so'. 
    Contesto:\n{context}'''),
        ("human", "{question}")
        ])
    
    comparatore = vettori.as_retriever(
        # mmr = maximal marginal relevance
        search_type="mmr",
        # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 4})
    
    modello_llm = ChatOpenAI(
        model="gpt-5.4-nano",
        temperature=0.4,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    catena = (
        # All'inizio mettiamo un dizionario che serve a costruire 
        # la struttura che il prompt vuol in input
        # Il comparatore produce i documenti (es. k=4) e li passa alla formattazione
        # RunnablePassthrough() vuol dire:
        # quando arriverà un input → passalo così com’è
        # Dobbiamo fare così perché ancora l'input concreto non c'è!  
        {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
        # StrOutputParser() prende l’output del modello 
        # e lo traforma in una stringa semplice (senza aggiunta di info ecc.)

    if domanda_utente:
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)

    
    

