import os
import re
from pathlib import Path
import logging
from llama_index.core import Document
from llama_index.readers.json import JSONReader
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceWindowNodeParser, MarkdownNodeParser
from llama_index.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL


def load_data(directory_path):
    """Load data from directory."""
    documents1 = []

    for md_filename in os.listdir(directory_path):
        if not md_filename.lower().endswith(".md"):
            continue
        file_path = os.path.join(directory_path, md_filename)
        raw = open(file_path, encoding="utf-8").read()

        # This regex finds all occurrences of <!-- PAGE: N -->
        # Split into pages and captures the page number
        splits = re.split(r"\s*<!--\s*PAGE:\s*(\d+)\s*-->\s*", raw)
        it = iter(splits)
        for page_num, page_text in zip(it, it):
            # Build metadata
            meta = {
                "file_type": "markdown",
                "connector_family": Path(md_filename).stem.upper(),
                "source": Path(md_filename).stem,
                "absolute_path": os.path.abspath(file_path),
                "page_number": int(page_num),
            }
            documents1.append(Document(text=page_text, metadata=meta))

    lab_dir = os.path.join(directory_path, "lab")
    if not os.path.exists(lab_dir):
        lab_dir = os.path.abspath(
            os.path.join(directory_path, "..", "extracted_best", "lab")
        )

    documents3 = []
    if os.path.exists(lab_dir):
        for md_filename in os.listdir(lab_dir):
            if not md_filename.lower().endswith(".md"):
                continue
            file_path = os.path.join(lab_dir, md_filename)
            raw = open(file_path, encoding="utf-8").read()

            splits = re.split(r"\s*<!--\s*PAGE:\s*(\d+)\s*-->\s*", raw)
            it = iter(splits)
            for page_num, page_text in zip(it, it):
                meta = {
                    "file_type": "markdown",
                    "connector_family": Path(md_filename).stem.upper(),
                    "source": Path(md_filename).stem,
                    "absolute_path": os.path.abspath(file_path),
                    "page_number": int(page_num),
                }
                documents3.append(Document(text=page_text, metadata=meta))

    reader = JSONReader()
    json_docs = []
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".json"):
            continue
        file_path = os.path.join(directory_path, filename)
        for doc in reader.load_data(input_file=file_path, extra_info={}):
            abs_file_path = os.path.abspath(file_path)
            doc.metadata.update(
                {
                    "file_type": "json",
                    "connector_family": Path(file_path).stem.upper(),
                    "source": Path(file_path).stem,
                    "absolute_path": abs_file_path,
                }
            )
            json_docs.append(doc)
    # combine
    documents1.extend(json_docs)
    return documents1, documents3


def process_data(documents1, documents3):
    """Process data and create indices."""
    try:
        logging.info(f"Processing documents1: {len(documents1)} documents")
        logging.info(f"Processing documents3: {len(documents3)} documents")

        # SentenceWindowNodeParser
        sentencewindow_node_parser = SentenceWindowNodeParser(
            include_metadata=True, include_prev_next_rel=True, window_size=5
        )
        # Backup cause markdown is the format
        markdown_node_parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True,
        )

        logging.info("Parsing nodes from documents...")
        # Catalogue data
        nodes_sentencewindow = sentencewindow_node_parser.get_nodes_from_documents(
            documents1
        )
        logging.info(f"Generated {len(nodes_sentencewindow)} sentence window nodes")

        nodes_markdown_nodes = markdown_node_parser.get_nodes_from_documents(documents1)
        logging.info(f"Generated {len(nodes_markdown_nodes)} markdown nodes")

        nodes_markdown = nodes_sentencewindow + nodes_markdown_nodes
        logging.info(f"Total catalogue nodes: {len(nodes_markdown)}")

        # Labs data
        nodes_sentencewindow1 = sentencewindow_node_parser.get_nodes_from_documents(
            documents3
        )
        logging.info(
            f"Generated {len(nodes_sentencewindow1)} lab sentence window nodes"
        )

        nodes_markdown_nodes_lab = markdown_node_parser.get_nodes_from_documents(
            documents3
        )
        logging.info(f"Generated {len(nodes_markdown_nodes_lab)} lab markdown nodes")

        nodes_markdown_lab = nodes_sentencewindow1 + nodes_markdown_nodes_lab
        logging.info(f"Total lab nodes: {len(nodes_markdown_lab)}")

        # Settings
        logging.info("Initializing language model and embedding model...")
        Settings.llm = Ollama(
            model=OLLAMA_MODEL,
            temperature=0.0,
            num_ctx=8012,
            top_p=0.5,
            base_url=OLLAMA_BASE_URL,
        )
        Settings.embed_model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )

        logging.info("Creating storage contexts...")
        # Markdown: Create storage context and add documents
        storage_context_markdown = StorageContext.from_defaults()
        storage_context_markdown.docstore.add_documents(nodes_markdown)

        # LAB Markdown: Create storage context and add documents
        storage_context_markdown_lab = StorageContext.from_defaults()
        storage_context_markdown_lab.docstore.add_documents(nodes_markdown_lab)

        logging.info("Creating vector indices...")
        # Markdown: Create both indices
        vector_index_markdown = VectorStoreIndex(
            nodes_markdown,
            storage_context=storage_context_markdown,
            similarity_top_k=25,
            index_kwargs={
                "metric": "cosine",
                "normalize_embeddings": True,
                "hnsw": {
                    "max-links-per-node": 64,
                    "neighbors-to-explore-at-insert": 300,
                    "ef_construction": 400,
                },
            },
        )
        logging.info("Vector index for markdown created successfully")

        # Lab Markdown: Create both indices
        vector_index_markdown_lab = VectorStoreIndex(
            nodes_markdown_lab,
            storage_context=storage_context_markdown_lab,
            similarity_top_k=25,
            index_kwargs={
                "metric": "cosine",
                "normalize_embeddings": True,
                "hnsw": {
                    "max-links-per-node": 64,
                    "neighbors-to-explore-at-insert": 300,
                    "ef_construction": 400,
                },
            },
        )
        logging.info("Vector index for lab markdown created successfully")

        logging.info("Creating keyword indices...")
        # Keywords indices
        keyword_index_markdown = SimpleKeywordTableIndex(
            nodes_markdown, storage_context=storage_context_markdown, show_progress=True
        )
        logging.info("Keyword index for markdown created successfully")

        keyword_index_markdown_lab = SimpleKeywordTableIndex(
            nodes_markdown_lab,
            storage_context=storage_context_markdown_lab,
            show_progress=True,
        )
        logging.info("Keyword index for lab markdown created successfully")

        # Verify that all indices were created correctly
        if vector_index_markdown is None:
            logging.error("ERROR: vector_index_markdown failed to initialize")
        if keyword_index_markdown is None:
            logging.error("ERROR: keyword_index_markdown failed to initialize")
        if vector_index_markdown_lab is None:
            logging.error("ERROR: vector_index_markdown_lab failed to initialize")
        if keyword_index_markdown_lab is None:
            logging.error("ERROR: keyword_index_markdown_lab failed to initialize")

        logging.info("All indices created successfully")

        return (
            vector_index_markdown,
            keyword_index_markdown,
            vector_index_markdown_lab,
            keyword_index_markdown_lab,
        )

    except Exception as e:
        logging.error(f"Error in processing_data: {str(e)}")
        # Return None values to indicate failure
        return None, None, None, None
