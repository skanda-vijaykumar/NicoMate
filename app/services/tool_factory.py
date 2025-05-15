import logging
from typing import List
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from langchain.tools import BaseTool, Tool
from app.services.search import MultiSearchRetriever
from app.api.dependencies import (
    vector_index_markdown,
    keyword_index_markdown,
    vector_index_markdown_lab,
    keyword_index_markdown_lab,
)
from app.core.retriever import CustomRetriever
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from app.core.source_tracker import SourceTracker
from app.services.react_integration import ConnectorDimensionLangchainTool

class RankedNodesLogger:
    def __init__(self, reranker):
        self.reranker = reranker
        self.source_tracker = SourceTracker()
        super().__init__()

    def postprocess_nodes(self, nodes, query_bundle):
        # Pass nodes through the underlying reranker
        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

        # Log the reranked nodes
        logging.info("\n=== RERANKED NODES ===")
        logging.info(f"Showing {len(reranked_nodes)} nodes after reranking")

        # Add all sources to the global tracker
        self.source_tracker.add_sources_from_nodes(reranked_nodes)

        for i, node in enumerate(reranked_nodes):
            if hasattr(node, "node") and hasattr(node.node, "metadata"):
                source = node.node.metadata.get("source", "Unknown")
                family = node.node.metadata.get("connector_family", "Unknown")
                file_type = node.node.metadata.get("file_type", "Unknown")
                score = node.score if hasattr(node, "score") else "N/A"
                abs_path = node.node.metadata.get("absolute_path", "Unknown path")

                logging.info(
                    f"Node {i + 1}: {source} | Family: {family} | Type: {file_type} | Score: {score}"
                )
                logging.info(f"  Path: {abs_path}")

                # First few characters of content for context
                node_text = node.node.text.replace("\n", " ")
                if len(node_text) > 100:
                    node_text = node_text[:100] + "..."
                logging.info(f"  Content: {node_text}")

        logging.info("=== END OF RERANKED NODES ===\n")

        return reranked_nodes


def create_tools() -> List[BaseTool]:
    """Create tools for the agent to use."""
    from app.api.dependencies import (
        vector_index_markdown, 
        keyword_index_markdown, 
        vector_index_markdown_lab, 
        keyword_index_markdown_lab
    )
    
    tools = []

    try:
        # Print input indices types
        logging.info("\nCreating tools with indices:")
        logging.info(
            f"- vector_index_markdown: {type(vector_index_markdown).__name__ if vector_index_markdown is not None else 'None'}"
        )
        logging.info(
            f"- keyword_index_markdown: {type(keyword_index_markdown).__name__ if keyword_index_markdown is not None else 'None'}"
        )
        logging.info(
            f"- vector_index_markdown_lab: {type(vector_index_markdown_lab).__name__ if vector_index_markdown_lab is not None else 'None'}"
        )
        logging.info(
            f"- keyword_index_markdown_lab: {type(keyword_index_markdown_lab).__name__ if keyword_index_markdown_lab is not None else 'None'}"
        )

        # Only create retrievers if necessary indices are available
        if vector_index_markdown is not None and keyword_index_markdown is not None:
            logging.info("Creating markdown tools...")
            try:
                # Create retrievers for markdowns
                vector_retriever_markdown = VectorIndexRetriever(
                    index=vector_index_markdown,
                    similarity_top_k=30,
                    vector_store_kwargs={
                        "search_kwargs": {"search_type": "similarity", "k": 30}
                    },
                )
                keyword_retriever_markdown = KeywordTableSimpleRetriever(
                    index=keyword_index_markdown, similarity_top_k=25
                )
                hybrid_retriever_markdown = CustomRetriever(
                    vector_retriever=vector_retriever_markdown,
                    keyword_retriever=keyword_retriever_markdown,
                    mode="OR",
                )

                # Filters and retriever strategies
                base_reranker = FlagEmbeddingReranker(
                    model="BAAI/bge-reranker-large", top_n=15
                )
                # Wrap the reranker with logger
                reranker = RankedNodesLogger(base_reranker)
                response_synthesizer = get_response_synthesizer(
                    response_mode="compact_accumulate", verbose=True
                )
                hybrid_query_engine_markdown = RetrieverQueryEngine(
                    retriever=hybrid_retriever_markdown,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[reranker],
                )

                # Create tools list
                query_engine_tools_markdown = [
                    QueryEngineTool(
                        query_engine=hybrid_query_engine_markdown,
                        metadata=ToolMetadata(
                            name="Nicomatic_connector_catalogue",
                            description="""
                            A technical repository for Nicomatic products specifications and compatibility information. Use this as your PRIMARY tool for:
                                - Cable compatibility questions (e.g., "What cable goes with 30-1447-ZZ?")
                                - Finding accessories that match with specific connectors
                                - Pitch size
                                - All part number lookups and compatibility checks
                                - Connector specifications including temperature ratings, electrical properties, materials and more
                                - Any question about what "works with", "goes with", or is "compatible with" a part number
                            This tool contains comprehensive product information beyond just dimensions. For questions about connector compatibility, accessories, or cables, ALWAYS use this tool first.
                            When using this tool make sure that the input if needed will have connector name mentioned which user is referring to like AMM, CMM, DMM, EMM.
                            """),
                    )
                ]

                # Convert tools for langchain
                llamaindex_to_langchain_converted_tools_markdown = [
                    t.to_langchain_tool() for t in query_engine_tools_markdown
                ]
                tools.extend(llamaindex_to_langchain_converted_tools_markdown)
                logging.info("Added markdown catalog tools")
            except Exception as e:
                logging.error(f"Error creating markdown tools: {str(e)}")
        else:
            logging.info("Skipping markdown tools due to missing indices")

        if (
            vector_index_markdown_lab is not None
            and keyword_index_markdown_lab is not None
        ):
            logging.info("Creating lab tools...")
            try:
                # Create retrievers for lab files
                vector_retriever_markdown_lab = VectorIndexRetriever(
                    index=vector_index_markdown_lab,
                    similarity_top_k=30,
                    vector_store_kwargs={
                        "search_kwargs": {"search_type": "similarity", "k": 30}
                    },
                )
                keyword_retriever_markdown_lab = KeywordTableSimpleRetriever(
                    index=keyword_index_markdown_lab, similarity_top_k=25
                )
                hybrid_retriever_markdown_lab = CustomRetriever(
                    vector_retriever=vector_retriever_markdown_lab,
                    keyword_retriever=keyword_retriever_markdown_lab,
                    mode="OR",
                )

                base_reranker = FlagEmbeddingReranker(
                    model="BAAI/bge-reranker-large", top_n=15
                )
                lab_reranker = RankedNodesLogger(base_reranker)

                response_synthesizer = get_response_synthesizer(
                    response_mode="accumulate", verbose=True
                )

                hybrid_query_engine_markdown_lab = RetrieverQueryEngine(
                    retriever=hybrid_retriever_markdown_lab,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[lab_reranker],
                )

                query_engine_tools_markdown_lab = [
                    QueryEngineTool(
                        query_engine=hybrid_query_engine_markdown_lab,
                        metadata=ToolMetadata(
                            name="Nicomatic_lab_tests",
                            description="Use this tool to find information about all Nicomatic connector's lab tests. This tool has access to the all lab tests of humidity, durability, mating and unmating forces and a lot more for AMM, CMM, DMM, EMM. Input must be a clear sentence.",
                        ),
                    )
                ]

                llamaindex_to_langchain_converted_tools_markdown_lab = [
                    t.to_langchain_tool() for t in query_engine_tools_markdown_lab
                ]
                tools.extend(llamaindex_to_langchain_converted_tools_markdown_lab)
                logging.info("Added lab tools")
            except Exception as e:
                logging.error(f"Error creating lab tools: {str(e)}")
        else:
            logging.info("Skipping lab tools due to missing indices")

        # Add internet search tool
        def search_function(query: str):
            search = MultiSearchRetriever()
            return search._get_relevant_documents(query)

        search_tool_dict = Tool(
            name="Internet_tool",
            func=search_function,
            description="A general-purpose search capability that accesses external data sources beyond Nicomatic's internal documentation. This tool should only be used as a fallback when other specialized tools fail to provide relevant information, or for queries about industry trends, competitor's products, other companies, historical information, information on people, names or general technical concepts not specific to Nicomatic products. Always prioritize Nicomatic's specialized tools over this general search function for product-specific information. When using this tool, always include 'Nicomatic' in the search query unless the question is clearly about a non-Nicomatic topic.",
        )

        tools.append(search_tool_dict)
        logging.info("Added internet search tool")

        # Add connector dimension tool
        try:
            from app.config import EXTRACTED_DATA_DIR
            
            dimension_tool = ConnectorDimensionLangchainTool(EXTRACTED_DATA_DIR)
            tools.append(dimension_tool)
            logging.info("Added connector dimension tool")
        except Exception as e:
            logging.error(f"Error adding connector dimension tool: {str(e)}")

    except Exception as e:
        logging.error(f"Error creating tools: {str(e)}")

        # At minimum, always add the search tool for fallback capability
        def search_function(query: str):
            search = MultiSearchRetriever()
            return search._get_relevant_documents(query)

        search_tool_dict = Tool(
            name="Internet_tool",
            func=search_function,
            description="A general-purpose search capability that accesses external data sources.",
        )
        tools.append(search_tool_dict)
        logging.info("Added internet search tool as fallback")

    logging.info(f"Created {len(tools)} tools")
    return tools
