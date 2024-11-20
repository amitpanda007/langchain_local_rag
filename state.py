import operator
import json
import os
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from vector_store import get_retriever
from langchain_community.tools.tavily_search import TavilySearchResults

local_llm = "llama3.2"
llm = ChatOllama(model=local_llm, temperature=0, base_url="http://localhost:11434/")
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json", base_url="http://localhost:11434/")

os.environ["TAVILY_API_KEY"] = "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"
web_search_tool = TavilySearchResults(k=3)

CHROMA_PATH = "chroma"  # Path to the directory to save Chroma database


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    retriever = get_retriever()
    documents = retriever.invoke(question)
    print(f"Found {len(documents)} documents from vector store")
    print(documents)
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    rag_prompt = """You are an assistant for question-answering tasks. 

        Here is the context to use to answer the question:
        
        {context} 
        
        Think carefully about the above context. 
        
        Now, review the user question:
        
        {question}
        
        Provide an answer to this questions using only the above context. 
        
        Use three sentences maximum and keep the answer concise.
        
        Answer:"""

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    print(generation.content)
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    # Doc grader instructions
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
    
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    # Grader prompt
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
    
    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
    
    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    do_web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            do_web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": do_web_search}


def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
    
    The vectorstore contains documents related to AI.
    
    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
    
    Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

    print("---ROUTE QUESTION---")
    _route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(_route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    # Hallucination grader instructions
    hallucination_grader_instructions = """
    
    You are a teacher grading a quiz. 
    
    You will be given FACTS and a STUDENT ANSWER. 
    
    Here is the grade criteria to follow:
    
    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
    
    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
    
    Score:
    
    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
    
    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
    
    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
    
    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
    
    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

    # Answer grader instructions
    answer_grader_instructions = """You are a teacher grading a quiz. 
    
    You will be given a QUESTION and a STUDENT ANSWER. 
    
    Here is the grade criteria to follow:
    
    (1) The STUDENT ANSWER helps to answer the QUESTION
    
    Score:
    
    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
    
    The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
    
    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
    
    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
    
    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 
    
    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"
