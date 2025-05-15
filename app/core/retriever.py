from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core import Document
import logging


class CustomRetriever(BaseRetriever):
    """Custom retriever that combines vector and keyword search with filtering."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _extract_connector_families(self, query_text: str) -> List[str]:
        """Extract mentioned connector families from the user query."""
        mentioned_families = []
        query_upper = query_text.upper()
        valid_families = ["AMM", "CMM", "DMM", "EMM", "DBM", "DFM"]

        for family in valid_families:
            if family in query_upper:
                mentioned_families.append(family)
        return mentioned_families

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant nodes based on query."""
        # Extract mentioned connector families from vdb
        mentioned_families = self._extract_connector_families(query_bundle.query_str)
        logging.info(f"Connector families mentioned in query: {mentioned_families}")

        # Get metadata from query bundle similarity nodes
        metadata = getattr(query_bundle, "extra_info", {}) or {}
        file_type = metadata.get("type")

        # Get basic retrievals
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        logging.info(f"Vector nodes retrieved: {len(vector_nodes)}")
        if vector_nodes:
            logging.info(f"First vector node: {vector_nodes[0].node.text[:100]}...")

        # Add try-except block here
        try:
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
            logging.info(f"Keyword nodes retrieved: {len(keyword_nodes)}")
        except AttributeError as e:
            logging.error(f"Error retrieving keyword nodes: {str(e)}")
            # If error empty list for keyword nodes
            keyword_nodes = []

        # Only apply connector family filtering if families are mentioned
        if mentioned_families:
            logging.info(f"Filtering by connector families: {mentioned_families}")
            # For vector nodes
            filtered_vector_nodes = []
            for node in vector_nodes:
                node_family = node.node.metadata.get("connector_family", "")
                # Check if node family starts with any of the mentioned families in metadata
                if any(
                    node_family.upper().startswith(family)
                    for family in mentioned_families
                ):
                    filtered_vector_nodes.append(node)

            # For key-words
            filtered_keyword_nodes = []
            for node in keyword_nodes:
                node_family = node.node.metadata.get("connector_family", "")
                # Check if node starts with any of the mentioned families in metadata
                if any(
                    node_family.upper().startswith(family)
                    for family in mentioned_families
                ):
                    filtered_keyword_nodes.append(node)

            # Combine both key words and vector nodes
            vector_nodes = filtered_vector_nodes
            keyword_nodes = filtered_keyword_nodes
            logging.info(
                f"Nodes after connector family filter: {len(vector_nodes)} vectors, {len(keyword_nodes)} keywords"
            )
            combined_nodes = vector_nodes + keyword_nodes
            represented_families = set()

            # Filter again after combining just to make sure that overlapped nodes are still relevant
            for node in combined_nodes:
                node_family = node.node.metadata.get("connector_family", "").upper()
                for family in mentioned_families:
                    if node_family.startswith(family):
                        represented_families.add(family)

            # Find missing families (incomplete will work later..)
            missing_families = set(mentioned_families) - represented_families

            # For each missing family, explicitly search for nodes
            if missing_families:
                logging.info(
                    f"Ensuring representation for missing families: {missing_families}"
                )
                # Test sequence
                for missing_family in missing_families:
                    family_query = f"{missing_family} temperature"
                    family_bundle = QueryBundle(family_query)
                    family_nodes = self._vector_retriever.retrieve(family_bundle)
                    # Filter and add top nodes for selected family
                    for node in family_nodes:
                        node_family = node.node.metadata.get(
                            "connector_family", ""
                        ).upper()
                        if node_family.startswith(missing_family):
                            logging.info(f"Adding node for {missing_family}")
                            vector_nodes.append(node)
                            break

        # Apply file type filtering if needed
        if file_type:
            vector_nodes = [
                n for n in vector_nodes if n.node.metadata.get("file_type") == file_type
            ]
            keyword_nodes = [
                n
                for n in keyword_nodes
                if n.node.metadata.get("file_type") == file_type
            ]
            logging.info(
                f"Nodes after file type filter: {len(vector_nodes)} vectors, {len(keyword_nodes)} keywords"
            )

        # Combine results based on mode (AND/OR)
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {}
        for n in vector_nodes:
            combined_dict[n.node.node_id] = n
        for n in keyword_nodes:
            combined_dict[n.node.node_id] = n

        # AND is for intersection between keywords and vectors
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        # OR for union between keywords and vectors
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        # Get final nodes
        retrieve_nodes = [
            combined_dict[rid] for rid in retrieve_ids if rid in combined_dict
        ]

        # Fallback if no nodes were retrieved
        if not retrieve_nodes:
            logging.info("No nodes retrieved after filtering, using fallback")
            if vector_nodes:
                return vector_nodes
            elif keyword_nodes:
                return keyword_nodes
            else:
                return []

        logging.info(f"Final nodes to return before reranking: {len(retrieve_nodes)}")
        return retrieve_nodes

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for the query."""
        query_bundle = QueryBundle(query)
        nodes = self._retrieve(query_bundle)
        documents = []

        for node in nodes:
            try:
                doc = Document(page_content=node.node.text, metadata=node.node.metadata)
                documents.append(doc)
            except AttributeError:
                continue

        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of _get_relevant_documents."""
        return self._get_relevant_documents(query)
