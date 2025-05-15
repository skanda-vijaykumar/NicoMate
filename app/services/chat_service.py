import logging
from langchain_core.messages import AIMessage
from app.api.dependencies import return_agent
from app.core.source_tracker import SourceTracker


async def generate_response(user_input, formatted_chat_history, agent, session_history):
    """Generate response for general chat route."""
    try:
        # Reset the source tracker for this query
        source_tracker = SourceTracker()
        source_tracker.reset()

        response = await agent.ainvoke(
            {"input": user_input, "chat_history": formatted_chat_history}
        )

        final_answer = ""
        intermediate_data = []
        used_tools = []

        if isinstance(response, dict):
            # Extract intermediate steps if available
            intermediate_steps = response.get("intermediate_steps", [])

            # Process intermediate steps
            for step_idx, step in enumerate(intermediate_steps):
                if len(step) > 1:
                    tool_name = (
                        step[0].tool if hasattr(step[0], "tool") else "Unknown tool"
                    )
                    tool_input = (
                        step[0].tool_input
                        if hasattr(step[0], "tool_input")
                        else "Unknown input"
                    )

                    # Track used tools
                    used_tools.append(tool_name)

                    # Get the actual output content
                    tool_output = step[1]
                    if hasattr(tool_output, "return_values"):
                        output_str = str(tool_output.return_values.get("output", ""))
                    else:
                        output_str = str(tool_output)

                    # Store output for response synthesis
                    intermediate_data.append(output_str)

            # Generate response
            from langchain_ollama import ChatOllama
            from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
                disable_streaming=False,
                base_url=OLLAMA_BASE_URL,
                num_ctx=8152,
                top_p=0.7,
                top_k=30,
                cache=False,
            )

            if (
                response.get("output", "")
                == "Agent stopped due to iteration limit or time limit."
            ):
                llm_prompt = f"""
                Please provide a helpful response to the user's question strictly based on the information gathered from the system.
                Response should be relevant and must answer user's question accurately.
                \nuser's question: {user_input}

                \nHere is information gathered from our systems:
                \n{''.join(intermediate_data)}
                
                \nconversation history just for context:
                \n{formatted_chat_history}
                
                """

                synthesized_response = await llm.ainvoke(llm_prompt)
                final_answer = str(synthesized_response.content).strip("{}")
            else:
                final_answer = response.get("output", "") or str(response)

            # Clean up final answer
            final_answer = final_answer.replace("Final Answer:", "").strip()

            # Create clickable file links if catalog or lab tools were used
            if (
                "Nicomatic_connector_catalogue" in used_tools
                or "Nicomatic_lab_tests" in used_tools
            ):
                # Get the list of NodeWithScore objects
                source_nodes = source_tracker.get_source_nodes()
                if source_nodes:
                    links_text = "\n\nView source documentation:"
                    link_parts = set()
                    for node_with_score in source_nodes:
                        # Access metadata from the inner .node attribute
                        if hasattr(node_with_score, "node") and hasattr(
                            node_with_score.node, "metadata"
                        ):
                            metadata = node_with_score.node.metadata
                            abs_path = metadata.get("absolute_path")
                            if not abs_path:
                                continue

                            filename = os.path.basename(abs_path)
                            file_base = os.path.splitext(filename)[0]
                            # convert .md → .pdf in the URL
                            import urllib.parse

                            encoded_path = urllib.parse.quote(abs_path).replace(
                                ".md", ".pdf"
                            )

                            # pull the page number out of the metadata (default to 1)
                            page_number = metadata.get("page_number", 1)

                            # Build the link
                            link = f"\n- {file_base}: http://172.30.2.97:8000/source_document/{encoded_path}#page={page_number}"
                            link_parts.add(link)

                    if link_parts:
                        # Sort links alphabetically for consistent order
                        links_text += "\n" + "\n".join(sorted(list(link_parts)))

                    # Combine answer with links
                    full_response = final_answer + links_text
                else:
                    full_response = final_answer
            else:
                full_response = final_answer

            session_history.add_message(AIMessage(content=full_response))
            yield full_response
        else:
            error_msg = "Received unexpected response format from agent"
            logging.error(error_msg)
            session_history.add_message(AIMessage(content=error_msg))
            yield error_msg

        return_agent(agent)
    except Exception as e:
        error_message = f"I apologize, but I encountered an error while processing your request: {str(e)}"
        logging.error(f"Error in generate_response: {str(e)}")
        session_history.add_message(AIMessage(content=error_message))
        yield error_message
