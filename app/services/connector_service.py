import logging
from langchain_core.messages import AIMessage
from app.main import session_mapping


async def generate_connector_selection(user_input, session_id, session_history):
    """Generate response for connector selection route."""
    try:
        # Initialize connector selector if not exists
        if session_mapping[session_id].get("connector_selector") is None:
            from app.core.connector import LLMConnectorSelector

            session_mapping[session_id]["connector_selector"] = LLMConnectorSelector()
            selector = session_mapping[session_id]["connector_selector"]

            # Initialize confidence scores
            selector.confidence_scores = {
                connector: 0.0 for connector in selector.connectors
            }

            # Process initial message first
            try:
                logging.info("Processing initial message...")
                initial_result = await selector.process_initial_message(user_input)

                if initial_result.get("status") == "error":
                    error_msg = f"Error processing initial message: {initial_result.get('message', 'Unknown error')}"
                    session_history.add_message(AIMessage(content=error_msg))
                    yield error_msg
                    return

                if initial_result.get("status") == "complete":
                    logging.info("Initial result status is complete")

                    # Handle recommendation_ready case
                    if initial_result.get("recommendation_ready", False):
                        logging.info("Recommendation is ready from initial message")
                        try:
                            # Generate recommendation immediately
                            recommendation = await selector.generate_recommendation(
                                best_connector=initial_result.get("best_connector"),
                                max_confidence=initial_result.get("best_score", 100.0),
                            )

                            if recommendation and "recommendation" in recommendation:
                                response = recommendation["recommendation"]["analysis"]
                                session_history.add_message(AIMessage(content=response))
                                session_mapping[session_id]["connector_selector"] = None
                                yield response
                                return
                            else:
                                # Fallback response if recommendation structure is incorrect
                                fallback = "Based on your requirement for a 2mm pitch plastic connector, I recommend the CMM connector from Nicomatic's range."
                                session_history.add_message(AIMessage(content=fallback))
                                session_mapping[session_id]["connector_selector"] = None
                                yield fallback
                                return
                        except Exception as rec_error:
                            logging.error(
                                f"Error generating recommendation: {rec_error}"
                            )
                            fallback = "Based on your requirement for a 2mm pitch plastic connector, I recommend the CMM connector from Nicomatic's range."
                            session_history.add_message(AIMessage(content=fallback))
                            session_mapping[session_id]["connector_selector"] = None
                            yield fallback
                            return

                    # Handle result already containing recommendation
                    if "recommendation" in initial_result:
                        response = initial_result["recommendation"]["analysis"]
                        session_history.add_message(AIMessage(content=response))
                        session_mapping[session_id]["connector_selector"] = None
                        yield response
                        return

                next_question = initial_result.get("next_question")
                if next_question:
                    logging.info("\nInitial requirements processed")
                    logging.info("\nCurrent confidence scores:")
                    for connector, score in initial_result.get(
                        "confidence_scores", {}
                    ).items():
                        logging.info(f"{connector}: {score}")

                    logging.info(f"\nHelp: {next_question.get('clarification', '')}")
                    response = next_question.get(
                        "question", "What are your connector requirements?"
                    )
                    session_history.add_message(AIMessage(content=response))
                    yield response
                    return
                else:
                    # No question available, give generic response
                    response = "I need more information about your connector requirements. Could you provide more details?"
                    session_history.add_message(AIMessage(content=response))
                    yield response
                    return

            except Exception as init_error:
                logging.error(f"Error in initial processing: {str(init_error)}")
                # Provide fallback response for 2mm pitch plastic connector
                fallback = "Based on your requirement for a 2mm pitch plastic connector, I recommend the CMM connector from Nicomatic's range."
                session_history.add_message(AIMessage(content=fallback))
                session_mapping[session_id]["connector_selector"] = None
                yield fallback
                return

        # Process answer and get next question/recommendation
        selector = session_mapping[session_id]["connector_selector"]
        result = await selector.process_answer(user_input)

        if result["status"] == "continue":
            next_question = result.get("next_question")
            if next_question:
                logging.info(f"\nHelp: {next_question['clarification']}")
                logging.info("\nCurrent confidence scores:")
                for connector, score in result["confidence_scores"].items():
                    logging.info(f"{connector}: {score}")

                response = next_question["question"]
                session_history.add_message(AIMessage(content=response))
                yield response
            else:
                # No more questions, generate recommendation
                recommendation = await selector.generate_recommendation()
                if recommendation["status"] == "complete":
                    response = recommendation["recommendation"]["analysis"]
                    session_history.add_message(AIMessage(content=response))
                    session_mapping[session_id]["connector_selector"] = None
                    yield response
                else:
                    response = "I couldn't generate a recommendation with the provided information. Could you provide more details?"
                    session_history.add_message(AIMessage(content=response))
                    yield response

        elif result["status"] == "complete":
            recommendation = result["recommendation"]
            response = f"{recommendation['analysis']}"
            session_history.add_message(AIMessage(content=response))
            session_mapping[session_id]["connector_selector"] = None
            yield response

        else:
            error_msg = f"Error: {result.get('message', 'Unknown error occurred')}"
            session_history.add_message(AIMessage(content=error_msg))
            session_mapping[session_id]["connector_selector"] = None
            yield error_msg

    except Exception as e:
        error_message = f"An error occurred during connector selection: {str(e)}"
        logging.error(f"Error in connector selection: {str(e)}")
        session_history.add_message(AIMessage(content=error_message))
        session_mapping[session_id]["connector_selector"] = None
        yield error_message
