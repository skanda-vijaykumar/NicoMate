import re
import logging
import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


async def determine_route(user_input, formatted_chat_history):
    """Determine whether to route to general or selection path."""

    # Direct routing based on patterns
    def pre_process_routing(user_input, formatted_chat_history):
        # Force routing to general for any input with a question mark
        if "?" in user_input:
            logging.info(
                "Direct routing enforcement: Input contains question mark, routing to general"
            )
            return {"score": "general"}

        # Check for WH-question words about connector properties
        wh_words = ["what", "where", "why", "who", "how", "which", "when"]
        if any(word in user_input.lower() for word in wh_words):
            logging.info(
                "Direct routing enforcement: Input contains WH-question words, routing to general"
            )
            return {"score": "general"}

        # Split the chat history to get the last message from the assistant
        messages = formatted_chat_history.split("\n")
        assistant_messages = [m for m in messages if m.startswith("AI:")]

        if assistant_messages:
            last_assistant_message = assistant_messages[-1][4:]  # Remove 'AI: ' prefix

            # Check if the last message was a recommendation
            connector_names = ["AMM", "CMM", "DMM", "EMM"]
            recommended_connectors = [
                c for c in connector_names if c in last_assistant_message
            ]

            # If the last message contained a connector recommendation
            if recommended_connectors and any(
                p in user_input.lower()
                for p in ["it", "this connector", "the connector"]
            ):
                logging.info(
                    "Direct routing enforcement: Question about recommended connector, routing to general"
                )
                return {"score": "general"}

        # Let the LLM handle other cases
        return None

    # Check pre-processing first
    pre_processed_route = pre_process_routing(user_input, formatted_chat_history)
    if pre_processed_route is not None:
        return pre_processed_route["score"]

    # Use LLM for more complex routing decisions
    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
            num_ctx=8152,
            cache=False,
            base_url=OLLAMA_BASE_URL,
            format="json",
        )

        prompt = PromptTemplate(
            template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a precise query routing AI. Your job is to analyze the conversation CONTEXT (`history`) and the user's INTENT in their latest input (`question`) to correctly route the query to either the 'selection' or 'general' channel.

    ### [Routing Channels:]

    1.  **selection**:
        Route here ONLY when the user's primary goal is **discovering or narrowing down the *type* of connector needed** based on requirements, **AND they have NOT mentioned a specific connector family (AMM, CMM, DMM, EMM) in their current input.** This includes when the user is:
        - Starting a search without a specific product line in mind (e.g., "I need a connector", "Help me find a connector for high power and signal").
        - Actively providing specifications or constraints ***specifically in response to a system question aimed at finding a suitable connector TYPE*** during an active selection process. Examples of such responses include values like "1.27mm", "20 pins", "metal housing", "EMI shielding required", "on board", "pcb to cable", "100 degrees C", "as small as possible", "no preference", "yes [to a selection question]", "I don't know".
        - Asking clarifying questions *about the selection process itself* or about a specification the system asked for, *while actively engaged in the selection flow* (e.g., "What does pitch size mean here?", "What temperature ranges are typical?", "What do you mean by operational current?", "Can you explain the housing options?").

    2.  **general**:
        Route here when the user's query **mentions a specific, named connector family (AMM, CMM, DMM, EMM)**, OR when they are seeking general information, definitions, discussing past recommendations, or providing conversational input not directly part of active selection. This includes:
        - **ABSOLUTE PRIORITY TRIGGER:** User input contains "AMM", "CMM", "DMM", or "EMM". **Route here REGARDLESS of context or any other words like 'need', 'require', or specifications mentioned.** This rule overrides ALL other signals.
            - **Examples:** "I need a DMM with 100 pins.", "Tell me about high-temp CMM options.", "What is the current rating for AMM?", "Compare DMM and CMM.", "Okay, what about the DMM?", "Information on EMM needed."
        - Seeking general information, definitions, or explanations about connectors or related concepts ***outside*** of an active system-led selection process OR when the question is *about* a specific named family (AMM, CMM, DMM, EMM). (e.g., "What is EMI shielding?", "Explain connector pitch.", "What's the standard pitch for DMM?").
        - Queries that commonly use WH-questions (What, Where, Why, How) or end in '?', asking for information rather than defining selection criteria. ***Crucially:*** A clarifying question *during* active `selection` (System: 'What pitch?', User: 'What does pitch mean?') stays in `selection`. A general knowledge question ("What is pitch?") goes to `general`.
        - Queries related to cables associated with connectors.
        - **Greetings, expressions of appreciation, or simple conversational filler** (e.g., "hello", "hi", "thanks", "thank you", "ok", "got it", "sounds good"). Route these directly to `general`.
        - Questions about connectors *previously recommended* by the system (e.g., "What's the lead time for that CMM you suggested?", "Tell me more about the DMM option.").

    ### [Core Guidelines:]

    1.  **SPECIFIC NAME PRECEDENCE (MOST IMPORTANT RULE):** If the user's latest input contains "AMM", "CMM", "DMM", or "EMM", the query **MUST** be routed to **general**. This rule takes absolute precedence over context, intent keywords (like 'need', 'looking for'), or the presence of specifications (like '100 pins', 'high temp').
    2.  **ACTIVE SELECTION CONTEXT OVERRIDES KEYWORDS:** If the `history` shows the system just asked a question to narrow down the connector *type* (e.g., asking about pins, pitch, temp, housing, shielding, mounting, connection type), the user's *direct answer* to that question **MUST** be routed to **selection**, UNLESS the answer itself contains 'AMM', 'CMM', 'DMM', or 'EMM' (Rule 1 applies). Do *not* route to `general` just because the answer contains words like 'need', 'require', 'information', or technical terms (like 'EMI shielding') if it's clearly answering a selection question in context.
    3.  **CONTEXT IS KING (Beyond Rule 2):** Analyze the full `history`. A user response like "100 degrees" is `selection` if the system just asked about temperature *during selection*. It's `general` if discussing a specific DMM's specs or asking a general knowledge question unrelated to active selection.
    4.  **INTENT MATTERS (Secondary to Name Precedence & Context):**
        - **Selection Intent:** User is trying to *figure out which type* of connector fits their abstract needs, *without having a specific family in mind*, and is actively participating in the system-led discovery process.
        - **General Intent:** User is asking *about* a known connector type/concept, providing conversational filler, or continuing a non-selection discussion.
    5.  **POST-RECOMMENDATION QUESTIONS (CRITICAL RULE):** Once the system has suggested a specific connector type or family (e.g., 'Based on your needs, a CMM connector seems suitable.'), any subsequent user questions asking *about the features, capabilities, or specifics of that suggested connector* (even if referred to as 'it' or 'that one') **MUST** go to **general**. Examples: 'What's the lead time for that CMM?', 'Can it handle 150 degrees?', 'Is it available in a right-angle version?', **'Can I mount it on a panel?'**. Route to `selection` again only if the user explicitly rejects the suggestion and wants to restart the search OR modifies their core requirements significantly.
    6.  **AMBIGUITY & CLARIFICATION:**
        - Vague answers ("yes", "ok", "I don't know", "no preference") during `selection` stay in `selection` as they are responses within that flow.
        - User questions ("what do you mean?", "?", "what is X?") should be routed based on context: if clarifying a `selection` question -> `selection`; if asking for general info or about a specific family -> `general`.

    ### [Output:]

    Provide your routing decision as a JSON response with a single key `'score'`. The value must be exactly `"general"` or `"selection"`.
    **Output *only* the raw JSON object.** No other text, comments, explanations, or markdown formatting are allowed.

    Example valid outputs:
    {{'score': 'selection'}}
    {{'score': 'general'}}

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the conversation history: \n\n {history} \n\n
    Here is the human input: {question}

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
            input_variables=["question", "history"],
        )

        retrieval_grader = prompt | llm | JsonOutputParser()

        routing_result = await retrieval_grader.invoke(
            {"question": user_input, "history": formatted_chat_history}
        )
        logging.info(f"LLM routing result: {routing_result}")

        if not isinstance(routing_result, dict) or "score" not in routing_result:
            # Fallback to general routing if response is invalid
            routing_result = {"score": "general"}
            logging.warning(
                f"Invalid routing response, falling back to general. Response: {routing_result}"
            )

        route = routing_result["score"]

        # Validate route value
        if route not in ["selection", "general"]:
            route = "general"
            logging.warning(f"Invalid route value: {route}, falling back to general")

        return route

    except Exception as e:
        logging.error(f"Error in route determination: {str(e)}")
        # Default to general on error
        return "general"
