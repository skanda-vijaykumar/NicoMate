from typing import List
import logging
from llama_index.core.schema import NodeWithScore

class SourceTracker:
    """Track sources for response generation."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SourceTracker, cls).__new__(cls)
            # Initialize with an empty list to store NodeWithScore objects
            cls._instance.nodes = []
        return cls._instance

    def reset(self):
        """Reset the list of nodes."""
        self.nodes = []

    def add_sources_from_nodes(self, nodes: List[NodeWithScore]):
        """Add sources from nodes."""
        for node in nodes:
            # Append the whole NodeWithScore object
            self.nodes.append(node)

    def get_source_nodes(self) -> List[NodeWithScore]:
        """Get the stored list of NodeWithScore objects."""
        return self.nodes

    def get_source_text(self) -> str:
        """Get text representation of sources."""
        if not self.nodes:
            return ""

        sources_list = []
        seen_sources = set()
        for node in self.nodes:
            # Access metadata from the inner node object
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                source = node.node.metadata.get("source", "Unknown")
                family = node.node.metadata.get("connector_family", "Unknown")
                # Avoid duplicate source/family pairs in the summary text
                if source != "Unknown" and (source, family) not in seen_sources:
                     sources_list.append(f"{source} ({family})")
                     seen_sources.add((source, family))

        if sources_list:
             return "\n\nSource documents: " + ", ".join(sorted(list(seen_sources), key=lambda x: x[0]))
        return ""

    def get_absolute_paths(self) -> List[str]:
        """Get absolute paths of source documents."""
        paths = set()
        for node in self.nodes:
             # Access metadata from the inner node object
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                path = node.node.metadata.get("absolute_path", None)
                if path:
                    paths.add(path)
        return list(paths)