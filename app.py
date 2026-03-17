import gradio as gr
from config import DB_PATH, EMBED_WORKERS
from ocr import ocr_status_message
from ingestor import ingest_pdfs, clear_all, file_list_md
from retriever import answer_question, export_chat

# ── Status strings shown in the sidebar ───────────────────────────────────────
_ocr_note     = ocr_status_message()
_persist_note = f"💾 Vector store: `{DB_PATH}` — embeddings persist across restarts."
_model_note   = f"🧠 LLM: `llama3.2:3b` · Embedding: `nomic-embed-text` · {EMBED_WORKERS} threads"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="PDF Q&A — llama3.2:3b") as demo:
    gr.Markdown(
        "## PDF Q&A Chatbot\n"
        "*Multi-PDF · Persistent store · Source chunks · Relevance scores · llama3.2:3b*"
    )

    with gr.Row():
        # ── Left panel — upload & status ──────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            pdf_upload = gr.File(
                label="Upload PDFs",
                file_types=[".pdf"],
                file_count="multiple",
            )
            ingest_btn = gr.Button("Ingest PDFs", variant="primary")
            clear_btn  = gr.Button("Clear all PDFs", variant="stop")
            file_list  = gr.Markdown(file_list_md())
            status_box = gr.Markdown("Upload one or more PDFs to get started.")
            gr.Markdown(_ocr_note)
            gr.Markdown(_persist_note)
            gr.Markdown(_model_note)

        # ── Right panel — chat ────────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=380)
            msg_box = gr.Textbox(
                placeholder="Ask anything about your PDFs… (Enter to send)",
                label="Your question",
                interactive=False,
                lines=2,
            )
            with gr.Row():
                send_btn       = gr.Button("Send", variant="primary")
                clear_chat_btn = gr.Button("Clear chat")
                export_btn     = gr.DownloadButton("⬇ Export chat", variant="secondary")

            with gr.Accordion("Retrieved source chunks & scores", open=False):
                sources_panel = gr.Markdown(
                    "*Ask a question to see retrieved chunks here.*"
                )

    # ── Event bindings ────────────────────────────────────────────────────────
    ingest_btn.click(
        ingest_pdfs,
        inputs=[pdf_upload],
        outputs=[status_box, msg_box, file_list],
    )

    clear_btn.click(
        clear_all,
        outputs=[chatbot, msg_box, status_box, msg_box, file_list],
    )

    send_btn.click(
        answer_question,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, msg_box, sources_panel],
    )

    msg_box.submit(
        answer_question,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, msg_box, sources_panel],
    )

    clear_chat_btn.click(
        lambda: ([], "", "*Ask a question to see retrieved chunks here.*"),
        outputs=[chatbot, msg_box, sources_panel],
    )

    export_btn.click(
        export_chat,
        inputs=[chatbot],
        outputs=[export_btn],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())