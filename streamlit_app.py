import asyncio
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# Environment
# -----------------------------

load_dotenv(
    Path(__file__).with_name(".env")
)


# -----------------------------
# Streamlit configuration
# -----------------------------

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered",
)


# -----------------------------
# Inngest client
# -----------------------------

@st.cache_resource
def get_inngest_client():

    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
    )


# -----------------------------
# Send ingestion event
# -----------------------------

async def send_ingest(
    pdf_path: str,
):

    client = get_inngest_client()

    ids = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": pdf_path,
                "source_id": Path(pdf_path).name,
            },
        )
    )

    return ids


# -----------------------------
# Send query event
# -----------------------------

async def send_query(
    question: str,
    top_k: int,
):

    client = get_inngest_client()

    ids = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return ids


# -----------------------------
# Wait for Inngest result
# -----------------------------

def wait_for_result(
    event_id: str,
    timeout: int = 120,
):

    url = (
        f"http://127.0.0.1:8288"
        f"/v1/events/{event_id}/runs"
    )

    start_time = time.time()

    while time.time() - start_time < timeout:

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        runs = data.get("data", [])

        if runs:

            run = runs[0]

            status = run.get("status")

            if status == "Completed":
                return run

            if status in [
                "Failed",
                "Cancelled",
            ]:

                raise RuntimeError(
                    f"Inngest function {status}"
                )

        time.sleep(1)

    raise TimeoutError(
        "Inngest function timed out."
    )


# ============================================================
# UI
# ============================================================

st.title(
    "📄 RAG PDF Assistant"
)

st.caption(
    "Upload a PDF and ask questions about its contents."
)


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📤 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)


if uploaded_file is not None:

    st.write(
        f"**Selected:** `{uploaded_file.name}`"
    )

    if st.button(
        "📥 Index PDF",
        use_container_width=True,
    ):

        try:

            # Create uploads directory
            upload_dir = Path("uploads")
            upload_dir.mkdir(
                exist_ok=True
            )

            # Save uploaded PDF
            pdf_path = (
                upload_dir /
                uploaded_file.name
            )

            with open(
                pdf_path,
                "wb",
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            st.info(
                "Sending PDF to the RAG ingestion pipeline..."
            )

            # Send ingestion event
            ids = asyncio.run(
                send_ingest(
                    str(pdf_path)
                )
            )

            event_id = ids[0]

            # Wait for ingestion
            with st.spinner(
                "Reading, chunking, embedding and indexing PDF..."
            ):

                run = wait_for_result(
                    event_id
                )

            output = run.get(
                "output",
                {}
            )

            ingested = output.get(
                "ingested",
                0
            )

            st.success(
                f"✅ PDF indexed successfully! "
                f"{ingested} chunks added to Qdrant."
            )

        except Exception as error:

            st.error(
                "❌ PDF ingestion failed."
            )

            st.exception(error)


# ============================================================
# DIVIDER
# ============================================================

st.divider()


# ============================================================
# QUERY
# ============================================================

st.header(
    "🔎 Ask a question"
)


with st.form(
    "query_form"
):

    question = st.text_input(
        "Your question",
        placeholder="What is stimulated emission?",
    )

    top_k = st.number_input(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button(
        "🔍 Ask",
        use_container_width=True,
    )


# ============================================================
# QUERY PROCESSING
# ============================================================

if submitted:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching your PDFs and generating an answer..."
            ):

                ids = asyncio.run(
                    send_query(
                        question.strip(),
                        int(top_k),
                    )
                )

                event_id = ids[0]

                run = wait_for_result(
                    event_id
                )

            output = run.get(
                "output",
                {}
            )

            # -------------------------
            # Answer
            # -------------------------

            st.success(
                "Answer generated successfully!"
            )

            st.subheader(
                "💬 Answer"
            )

            answer = output.get(
                "answer",
                "No answer returned.",
            )

            st.write(answer)

            # -------------------------
            # Sources
            # -------------------------

            sources = output.get(
                "sources",
                []
            )

            if sources:

                st.subheader(
                    "📚 Sources"
                )

                for source in sources:

                    st.write(
                        f"• {source}"
                    )

            # -------------------------
            # Context count
            # -------------------------

            num_contexts = output.get(
                "num_contexts"
            )

            if num_contexts is not None:

                st.caption(
                    f"Retrieved {num_contexts} context chunks."
                )

            # -------------------------
            # Debug
            # -------------------------

            with st.expander(
                "🔧 Run details"
            ):

                st.write(
                    "Event ID:",
                    event_id,
                )

                st.write(
                    "Run ID:",
                    run.get("run_id"),
                )

                st.write(
                    "Status:",
                    run.get("status"),
                )

        except Exception as error:

            st.error(
                "❌ RAG query failed."
            )

            st.exception(error)