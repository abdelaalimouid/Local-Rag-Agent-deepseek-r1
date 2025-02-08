import gradio as gr
from sentence_transformers import SentenceTransformer
from rag_module import load_faiss_index, query_ollama
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("Loading embedding model...")
embedder = SentenceTransformer("all-mpnet-base-v2", device="cpu")

def load_index(pdf_file):
    start_time = time.time()
    index, documents = load_faiss_index(embedder, pdf_file_path=pdf_file.name)
    logger.info(f"Index loaded in {time.time() - start_time:.2f}s")
    return index, documents

css = "footer {visibility: hidden} .small-text {font-size: 0.8em !important}"

with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="slate")) as app:
    gr.Markdown("# 📚 Agent RAG Local")
    gr.Markdown("Interrogez des documents PDF avec DeepSeek 1.5B", elem_classes=["small-text"])
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Déposer PDF", file_types=[".pdf"], file_count="single")
            with gr.Accordion("Paramètres", open=False):
                top_k = gr.Slider(1, 10, value=3, label="Nombre de contextes")
                temperature = gr.Slider(0.1, 1.0, value=0.7, label="Créativité")
        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Votre question", placeholder="Posez votre question ici...", lines=3)
            submit_btn = gr.Button("Envoyer", variant="primary")
            response_output = gr.Textbox(label="Réponse", interactive=False, show_copy_button=True)
            context_output = gr.JSON(label="Contextes utilisés", visible=True)
    
    examples = gr.Examples(
        examples=[["Quel est le sujet principal de ce document ?"],["Expliquez les concepts clés mentionnés"],["Résumez les conclusions principales"]],
        inputs=[query_input],
        label="Exemples de questions"
    )
    
    def process_query(query, pdf, k, t):
        try:
            index, documents = load_index(pdf)
            answer, context = query_ollama(query, embedder, index, documents, top_k=k, temperature=t)
            return answer, context
        except Exception as e:
            logger.exception("Error processing query")
            return f"Erreur: {str(e)}", []

    submit_btn.click(process_query, inputs=[query_input, pdf_input, top_k, temperature], outputs=[response_output, context_output])

if __name__ == "__main__":
    app.launch(server_port=7860, share=False)