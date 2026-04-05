
RESEARCHER_SYSTEM_PROMPT = """
You are a precise Product Knowledge Analyst. Your job is to extract relevant factual evidence from retrieved product documentation chunks and present it in a structured format.

CORE RULES:
1. Use ONLY information present in the provided context chunks.
2. If the context does not contain enough information to answer the query, explicitly state that, never fabricate details.
3. When multiple chunks contain related information, synthesise them coherently, do not repeat the same fact twice.
4. Cite which chunk number each fact comes from using [Chunk N] notation.
5. Do not answer the customer's question directly — your output is internal evidence for the Sales Closer agent.
6. Keep your output concise and factual. No marketing language.
7. Read each chunk, assess its relevance, then compile only what directly addresses the query.

OUTPUT STRUCTURE:

RELEVANT FACTS:
- [Chunk N] <fact>
- [Chunk N] <fact>
... (include all directly relevant facts)

GAPS:
<List any aspects of the query that the retrieved chunks do not address, or write "None" if the context fully covers the query.>
"""

def build_researcher_prompt(user_query: str, retrieved_chunks: list[str]) -> str:
    
    # Construct the full researcher prompt by injecting numbered chunks so the model can cite them precisely.
    if not retrieved_chunks:
        return (
            f"{RESEARCHER_SYSTEM_PROMPT}\n\n"
            f"CUSTOMER QUERY:\n{user_query}\n\n"
            f"RETRIEVED CONTEXT:\n[No relevant documents found in the knowledge base.]"
        )

    numbered_chunks = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.strip()}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    return (
        f"{RESEARCHER_SYSTEM_PROMPT}\n\n"
        f"CUSTOMER QUERY:\n{user_query}\n\n"
        f"RETRIEVED CONTEXT:\n{numbered_chunks}\n\n"
        f"Now extract and compile the relevant evidence following the output structure above."
    )


def agent_researcher(user_query: str, retriever, model=None) -> dict:
    # Retrieve relevant chunks and optionally synthesise them into structured evidence via the LLM.

    docs = retriever.invoke(user_query)

    if not docs:
        return {
            "raw_context": "",
            "synthesis": "No relevant product information found for this query.",
            "chunk_count": 0,
            "has_gaps": True,
        }

    chunks = [doc.page_content for doc in docs]
    raw_context = "\n\n---\n\n".join(chunks)

    if model is None:
        return {
            "raw_context": raw_context,
            "synthesis": raw_context,
            "chunk_count": len(chunks),
            "has_gaps": False,
        }

    prompt = build_researcher_prompt(user_query, chunks)
    synthesis = model.generate_content(prompt).text

    has_gaps = (
        "gaps:" in synthesis.lower()
        and "none" not in synthesis.lower().split("gaps:")[-1][:60]
    )

    return {
        "raw_context": raw_context,
        "synthesis": synthesis,
        "chunk_count": len(chunks),
        "has_gaps": has_gaps,
    }

