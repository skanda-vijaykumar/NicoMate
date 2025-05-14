import logging
import re
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from dateutil.parser import parse
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import Document
from app.config import TAVILY_API_KEY, SERPER_API_KEY
from tavily import TavilyClient
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, GoogleSerperAPIWrapper
from langchain_ollama import OllamaEmbeddings
from app.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL
from difflib import SequenceMatcher
import math
import time
from itertools import chain

class MultiSearchRetriever(BaseRetriever):
    """Retriever that searches multiple search engines."""
    
    def __init__(self):
        super().__init__()
        # Initialize search clients
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.serper_client = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
        self.ddg_search = DuckDuckGoSearchAPIWrapper()
        
        try:
            # Initialize embeddings
            self.encoder = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            self.encoder = None
            logging.info("Falling back to basic text similarity")

    def _retrieve(self, query_bundle):
        """Retrieve documents for the query."""
        query_str = query_bundle.query_str if hasattr(query_bundle, 'query_str') else str(query_bundle)
        docs = self._get_relevant_documents(query_str)
        nodes_with_scores = []
        for i, doc in enumerate(docs):
            from llama_index.core.schema import NodeWithScore, Node
            node = Node(text=doc.page_content, metadata=doc.metadata)
            score = 1.0 - (i * 0.1)
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores
    
    def _get_tavily_documents(self, query: str) -> List[Document]:
        """Get documents from Tavily search."""
        try:
            response = self.tavily_client.search(query=query)
            results = response.get("results", [])
            documents = []
            for res in results[:3]:
                if len(res.get("content", "")) < 50:
                    continue
                doc = Document(page_content=res.get("content", ""))
                doc.metadata = {
                    "source": res.get("url", ""),
                    "title": res.get("title", ""),
                    "provider": "Tavily",
                    "orig_order": len(documents)
                }
                documents.append(doc)
            return documents
        except Exception as e:
            logging.error(f"Error getting Tavily results: {e}")
            return []
        
    def _get_serper_documents(self, query: str) -> List[Document]:
        """Get documents from Google Serper."""
        try:
            raw_results = self.serper_client.results(query)
            results = raw_results.get("organic", [])
            documents = []
            for res in results[:3]:
                if len(res.get("snippet", "")) < 50:
                    continue
                doc = Document(page_content=res.get("snippet", ""))
                doc.metadata = {
                    "source": res.get("link", ""),
                    "title": res.get("title", ""),
                    "provider": "Google Serper",
                    "orig_order": len(documents)
                }
                documents.append(doc)
            return documents
        except Exception as e:
            logging.error(f"Error getting Serper results: {e}")
            return []
        
    def _get_ddg_documents(self, query: str) -> List[Document]:
        """Get documents from DuckDuckGo search."""
        try:
            time.sleep(1)  # Rate limiting
            results = self.ddg_search.run(query)
            if not results or len(results) < 50:
                return []
            
            doc = Document(page_content=results)
            doc.metadata = {
                "source": "DuckDuckGo Search",
                "title": "DuckDuckGo Results",
                "provider": "DuckDuckGo",
                "orig_order": 0
            }
            return [doc]
        except Exception as e:
            logging.error(f"Error getting DuckDuckGo results: {e}")
            return []
        
    def _extract_date(self, content: str) -> datetime:
        """Extract date from content."""
        date_patterns = [
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content.lower(), re.IGNORECASE)
            if matches:
                try:
                    return parse(matches[0])
                except (ValueError, TypeError):
                    continue
        return None
    
    def _compute_scores(self, query: str, documents: List[Document]) -> List[tuple]:
        """Compute relevance scores for documents."""
        if not documents:
            return []
            
        current_date = datetime.now()
        
        try:
            if self.encoder is not None:
                # Get text content from documents
                doc_texts = []
                for doc in documents:
                    if hasattr(doc, 'text_resource') and doc.text_resource:
                        doc_texts.append(doc.text_resource.text)
                    elif hasattr(doc, 'page_content'):
                        doc_texts.append(doc.page_content)
                    else:
                        continue

                # Get embeddings
                query_embedding = self.encoder.embed_query(query)
                doc_embeddings = self.encoder.embed_documents([doc.page_content for doc in documents])
                
                # Compute cosine similarities
                query_embedding = np.array(query_embedding)
                doc_embeddings = np.array(doc_embeddings)
                
                similarities = np.dot(doc_embeddings, query_embedding) / (
                    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
            else:
                # Fallback to simple text matching
                similarities = [
                    SequenceMatcher(None, query.lower(), doc.page_content.lower()).ratio() 
                    for doc in documents
                ]
                similarities = np.array(similarities)
            
            scored_docs = []
            for doc, similarity in zip(documents, similarities):
                doc_date = self._extract_date(doc.page_content)
                
                if doc_date:
                    days_old = (current_date - doc_date).days
                    recency_score = math.exp(-max(0, days_old) / 365)
                else:
                    recency_score = 0.5
                    
                # Combined score with weights for similarity and recency
                combined_score = (0.35 * float(similarity)) + (0.55 * recency_score)
                scored_docs.append((doc, combined_score))
                
                # Add scores to metadata
                doc.metadata["similarity_score"] = float(similarity)
                doc.metadata["recency_score"] = float(recency_score)
                doc.metadata["combined_score"] = float(combined_score)
            
            return sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logging.error(f"Error in computing scores: {e}")
            # Return original documents with neutral scores if scoring fails
            return list(zip(documents, [1.0] * len(documents)))
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from all search providers."""
        # Get results from all search providers
        tavily_docs = self._get_tavily_documents(query)
        serper_docs = self._get_serper_documents(query)
        ddg_docs = self._get_ddg_documents(query)
        
        # Combine all results
        all_docs = list(chain(tavily_docs, serper_docs, ddg_docs))
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_docs = []
        for doc in all_docs:
            url = doc.metadata["source"]
            if url not in seen_urls:
                seen_urls.add(url)
                unique_docs.append(doc)
        
        # Rerank documents based on combined score
        reranked_docs = self._compute_scores(query, unique_docs)
        
        # Return top 10 documents
        return [doc for doc, _ in reranked_docs[:10]]
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of _get_relevant_documents."""
        return self._get_relevant_documents(query)