"""
Podcast script writing module.

This module handles the generation of natural conversational scripts for podcasts.
It uses LLMs to create engaging question-answer exchanges between an interviewer
and interviewee based on podcast outlines and research material.

Key functionality includes:
- Converting outline sections into natural dialogue
- Maintaining conversational context and flow
- Formatting conversation history for prompt context
- Generating follow-up questions based on previous answers
- Structuring complete podcast scripts with proper speaker labels

The module leverages LangChain and GPT-4 to create dynamic multi-turn conversations
that sound natural while covering the key points from the outline. It includes
rate limiting and handles long-form content generation.

Example:
    script = write_podcast_script(
        config=podcast_config,
        outline=episode_outline,
        background_info=research_docs
    )
    print(script.as_str)
"""


import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from podcast_llm.outline import (
    format_wikipedia_document
)
import logging
from langchain import hub
from langchain_openai import ChatOpenAI
from podcast_llm.outline import PodcastOutline
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chains.llm import LLMChain
from langchain_core.vectorstores.base import VectorStoreRetriever
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.embeddings import get_embeddings_model
from podcast_llm.utils.llm import get_long_context_llm
from podcast_llm.models import (
    PodcastOutline,
    PodcastSection,
    PodcastSubsection,
    Script,
    Question,
    Answer
)
from podcast_llm.utils.rate_limits import retry_with_exponential_backoff


logger = logging.getLogger(__name__)


def format_conversation_history(conversation_history: list) -> str:
    """
    Format a conversation history into a readable string.

    Takes a list of Question and Answer objects representing a conversation history
    and formats them into a structured string with clear speaker labels. Each
    question is prefixed with "Interviewer:" and each answer with "Interviewee:".

    Args:
        conversation_history (list): List of alternating Question and Answer objects
            representing the conversation history

    Returns:
        str: Formatted string containing the full conversation with speaker labels
    """
    conversation = ""
    for c in conversation_history:
        if type(c) == Question:
            conversation += f"Interviewer: {c.as_str}\n"
        else:
            conversation += f"Interviewee: {c.as_str}\n"

    return conversation


def format_vector_results(docs: List[Document]):
    """
    Format retrieved vector store documents into a readable string.

    Takes a list of document objects returned from a vector store retrieval and 
    formats their content into a single string, with documents separated by newlines.
    This is used to format relevant background information for the LLM to reference
    when generating responses.

    Args:
        docs: List of document objects from vector store retrieval, each containing
            page_content attribute with the document text

    Returns:
        str: Concatenated document contents separated by double newlines
    """
    return "\n\n".join([d.page_content for d in docs])


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
def ask_question(topic: str, 
                 outline: PodcastOutline, 
                 section: PodcastSection, 
                 subsection: PodcastSubsection, 
                 background_info: list, 
                 draft_discussion: list, 
                 interviewer_chain: LLMChain) -> Question:
    """
    Generate the next interview question based on the conversation context.

    Uses LangChain and an LLM to generate a natural follow-up question that advances
    the discussion while staying focused on the current subsection topic. Takes into
    account the full conversation history and background research to ask relevant and
    insightful questions.

    Args:
        topic (str): The main podcast topic
        outline (PodcastOutline): The structured outline for the episode
        section (PodcastSection): The current section being discussed
        subsection (PodcastSubsection): The current subsection being discussed
        background_info (list): List of Wikipedia document objects with research material
        draft_discussion (list): List of previous Question and Answer objects
        interviewer_chain (LLMChain): The LangChain chain for generating questions

    Returns:
        Question: A structured Question object containing the generated question text
    """
    return interviewer_chain.invoke({
        'topic': topic,
        'outline': outline.as_str,
        'section': section.title,
        'subsection': subsection.title,
        'background_info': "\n\n".join([format_wikipedia_document(d) for d in background_info]),
        'conversation_history': format_conversation_history(draft_discussion)
    })


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
def answer_question(topic: str,
                    outline: PodcastOutline,
                    section: PodcastSection,
                    subsection: PodcastSubsection,
                    draft_discussion: list,
                    retriever: VectorStoreRetriever,
                    interviewee_chain: LLMChain) -> Answer:
    """
    Generate an answer to the current interview question.

    Uses LangChain and an LLM to generate a natural, informative response based on the 
    retrieved background information and conversation context. The response stays focused
    on the current subsection topic while maintaining a conversational tone.

    Args:
        topic (str): The main podcast topic
        outline (PodcastOutline): The structured outline for the episode
        section (PodcastSection): The current section being discussed 
        subsection (PodcastSubsection): The current subsection being discussed
        draft_discussion (list): List of previous Question and Answer objects
        retriever (VectorStoreRetriever): Retriever for getting relevant background info
        interviewee_chain (LLMChain): The LangChain chain for generating answers

    Returns:
        Answer: A structured Answer object containing the generated response text
    """
    background_information = format_vector_results(
        retriever.invoke(draft_discussion[-1].question))

    return interviewee_chain.invoke({
        'topic': topic,
        'outline': outline.as_str,
        'section': section.title,
        'subsection': subsection.title,
        'word_count': 100,
        'background_information': background_information,
        'conversation_history': format_conversation_history(draft_discussion),
        'question': draft_discussion[-1].as_str
    })


def discuss(config: PodcastConfig,
            topic: str, 
            outline: PodcastOutline, 
            background_info: List[Document], 
            vector_store: InMemoryVectorStore, 
            qa_rounds: int) -> list:
    """
    Simulate a podcast discussion through a series of questions and answers.

    Coordinates the generation of a natural-sounding podcast discussion by alternating
    between generating interview questions and detailed responses. Uses separate LLM chains
    for the interviewer and interviewee roles, with rate limiting to manage API usage.
    The discussion follows the podcast outline structure, exploring each subsection
    through multiple rounds of Q&A.

    Args:
        topic (str): The main podcast topic
        outline (PodcastOutline): Structured outline containing sections and subsections
        background_info (list): List of Wikipedia document objects with research material
        vector_store (InMemoryVectorStore): Vector store containing indexed research content
        qa_rounds (int): Number of question-answer rounds per subsection

    Returns:
        list: List of alternating Question and Answer objects forming the discussion
    """
    logger.info(f"Simulating discussion on: {topic}")

    interviewer_prompthub_path = "evandempsey/podcast_interviewer_role:bc03af97"
    interviewer_prompt = hub.pull(interviewer_prompthub_path)
    logger.info(f"Got prompt from hub: {interviewer_prompthub_path}")

    interviewee_prompthub_path = "evandempsey/podcast_interviewee_role:0832c140"
    interviewee_prompt = hub.pull(interviewee_prompthub_path)
    logger.info(f"Got prompt from hub: {interviewee_prompthub_path}")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.2,
        check_every_n_seconds=0.1,
        max_bucket_size=10
    )

    interviewer_llm = get_long_context_llm(config, rate_limiter)
    interviewee_llm = get_long_context_llm(config, rate_limiter)
    interviewer_chain = interviewer_prompt | interviewer_llm.with_structured_output(Question)
    interviewee_chain = interviewee_prompt | interviewee_llm.with_structured_output(Answer)

    retriever = vector_store.as_retriever(k=4)

    draft_discussion = []

    for section in outline.sections:
        for subsection in section.subsections:
            logger.info(f"Discussing section '{section.title}' subsection '{subsection.title}'")
            for _ in range(qa_rounds):
                draft_discussion.append(ask_question(
                    topic, 
                    outline, 
                    section,
                    subsection,
                    background_info, 
                    draft_discussion, 
                    interviewer_chain
                ))
                draft_discussion.append(answer_question(
                    topic,
                    outline,
                    section,
                    subsection,
                    draft_discussion,
                    retriever,
                    interviewee_chain
                ))

    return draft_discussion


def write_draft_script(config: PodcastConfig,
                       topic: str, 
                       outline: PodcastOutline, 
                       background_info: List[Document], 
                       deep_info: List[Document], 
                       qa_rounds: int):
    """
    Write a complete draft podcast script through simulated Q&A discussion.

    This function orchestrates the generation of a podcast script by:
    1. Creating a vector store from background research and deep dive articles
    2. Splitting content into manageable chunks for retrieval
    3. Simulating an interview-style discussion with alternating questions and answers
    
    Args:
        topic (str): The main topic of the podcast episode
        outline (PodcastOutline): Structured outline containing sections and subsections
        background_info (list): List of Wikipedia document objects with background research
        deep_info (list): List of specialized articles for in-depth discussion
        qa_rounds (int): Number of question-answer exchanges per subsection

    Returns:
        list: Alternating Question and Answer objects representing the complete discussion

    The function uses LangChain and GPT-4 to generate natural-sounding dialogue,
    with the vector store enabling relevant information retrieval for detailed responses.
    The resulting script follows the outline structure while maintaining conversational flow.
    """
    logger.info(f"Writing podcast script on: {topic}")

    logger.info("Creating vector store for documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Process background Wikipedia articles
    background_texts = []
    for doc in background_info:
        background_texts.append(doc.page_content)
    
    # Process deep research articles
    deep_texts = []
    for article in deep_info:
        deep_texts.append(article.page_content)

    # Combine all texts and split into chunks
    all_texts = background_texts + deep_texts
    chunks = text_splitter.create_documents(all_texts)

    # Create vector store
    embeddings = get_embeddings_model(config)
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    draft_script = discuss(config, topic, outline, background_info, vector_store, qa_rounds)
    return draft_script


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
def rewrite_script_section(section: list, rewriter_chain) -> list:
    """
    Rewrite a section of the podcast script to improve flow and naturalness.

    Takes a section of the draft script (a sequence of Question/Answer exchanges) and uses 
    the rewriter chain to improve the conversational flow, word choice, and overall quality
    while maintaining the core content and structure.

    Args:
        section (list): List of Question/Answer objects representing a script section
        rewriter_chain (LLMChain): Chain configured with prompt and model for rewriting

    Returns:
        list: List of dictionaries containing rewritten lines with structure:
            {
                'speaker': str,  # Speaker identifier ('Interviewer' or 'Interviewee')
                'text': str      # Rewritten line content
            }
    """
    rewritten = rewriter_chain.invoke({
        "script": format_conversation_history(section)
    })

    return [{'speaker': line.speaker, 'text': line.text} for line in rewritten.lines]


def write_final_script(config: PodcastConfig, topic: str, draft_script: list, batch_size: int = 4) -> list:
    """
    Rewrite a draft podcast script to improve flow, naturalness and quality.

    Takes a draft script consisting of Question/Answer exchanges and processes it in batches,
    using an LLM to improve the conversational flow, word choice, and overall quality while 
    maintaining the core content and structure. The script is processed in batches to manage
    context length and rate limits.

    Args:
        draft_script (list): List of Question/Answer objects representing the full draft script
        batch_size (int, optional): Number of Q/A exchanges to process in each batch. Defaults to 4.

    Returns:
        list: List of dictionaries containing the rewritten script lines with structure:
            {
                'speaker': str,  # Speaker identifier ('Interviewer' or 'Interviewee') 
                'text': str      # Rewritten line content
            }
    """
    logger.info("Processing draft script in batches")

    rewriter_prompthub_path = "evandempsey/podcast_rewriter:181421e2"
    rewriter_prompt = hub.pull(rewriter_prompthub_path)
    logger.info(f"Got prompt from hub: {rewriter_prompthub_path}")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.2,
        check_every_n_seconds=0.1,
        max_bucket_size=10
    )

    long_context_llm = get_long_context_llm(config, rate_limiter)
    long_context_llm = get_long_context_llm(config, rate_limiter)
    rewriter_chain = rewriter_prompt | long_context_llm.with_structured_output(Script)
    
    final_script = []
    
    # Process script in batches of bath_size
    for i in range(0, len(draft_script), batch_size):
        logger.info(f"Rewriting lines {i+1} to {i+batch_size} of {len(draft_script)}")
        batch = draft_script[i:i + batch_size]
        final_script.extend(rewrite_script_section(batch, rewriter_chain))

    # Add intro line
    final_script.insert(0, {
        'speaker': 'Interviewer',
        'text': config.intro.format(topic=topic, podcast_name=config.podcast_name)
    })

    # Add outro line
    final_script.append({
        'speaker': 'Interviewer',
        'text': config.outro.format(topic=topic, podcast_name=config.podcast_name)
    })
        
    return final_script
