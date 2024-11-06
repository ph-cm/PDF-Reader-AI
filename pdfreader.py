import fitz  # PyMuPDF para manipulação de PDFs
from transformers import pipeline
import nltk

# Configuração para usar nltk e baixar o pacote 'punkt' se necessário
nltk.download('punkt')
nltk.data.path.append("C:/Users/seu_usuario/nltk_data")  # Ajuste o caminho se necessário

# Função para extrair o texto do PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text("text")  # Extrai o texto em formato de string
    print("Texto extraído do PDF com sucesso.")
    return text

# Função para dividir o texto em blocos de sentenças maiores
def split_text_to_large_blocks(text, max_sentences=6):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        current_chunk += sentence + " "
        if (i + 1) % max_sentences == 0:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Texto dividido em {len(chunks)} blocos maiores.")
    return chunks

# Carrega o modelo de perguntas e respostas usando transformers
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Função principal para responder perguntas sobre o PDF
def answer_question_about_pdf(pdf_path, question):
    # Extrair e dividir o texto do PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text_to_large_blocks(pdf_text, max_sentences=6)

    # Variável para armazenar as respostas e similaridades
    respostas = []

    # Itera pelos chunks para encontrar respostas
    for i, chunk in enumerate(text_chunks):
        try:
            print(f"Processando bloco {i + 1}/{len(text_chunks)}")
            # Usa o modelo para responder à pergunta em cada parte do texto
            response = qa_pipeline(question=question, context=chunk)
            answer = response["answer"]

            # Filtra respostas curtas e verifica se a resposta é relevante
            if len(answer.split()) > 3:  # Ignora respostas muito curtas
                respostas.append(answer)

        except Exception as e:
            print(f"Erro ao processar o bloco {i + 1}: {e}")
    
    # Seleciona a resposta mais completa, se houver respostas
    if respostas:
        resposta_final = max(respostas, key=len)  # Escolhe a resposta mais longa
        return resposta_final
    else:
        return "Não foi possível encontrar uma resposta precisa para a pergunta."

# Exemplo de uso
pdf_path = "C:/Users/phenr/Downloads/Artificial Intelligence with Python.pdf"
question = "Provide a definition of artificial intelligence."
resposta = answer_question_about_pdf(pdf_path, question)
print("Resposta:", resposta)
