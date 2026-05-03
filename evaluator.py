from groq import Groq
import os
import json

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluate(query, chunks, answer):
    """
    Takes the query, retrieved chunks, and generated answer.
    Asks the LLM to score the result on 3 dimensions.
    Returns a dict with scores and reasoning.
    """

    context = "\n\n---\n\n".join(chunks)

    eval_prompt = f"""You are an expert evaluator for a RAG (Retrieval Augmented Generation) system.
Evaluate the following and return ONLY a JSON object, no extra text.

Query: {query}

Retrieved Context:
{context}

Generated Answer: {answer}

Score each dimension from 1 to 5:
- retrieval_relevance: Are the retrieved chunks relevant to the query?
- answer_grounding: Is the answer fully supported by the context (no hallucination)?
- answer_quality: Is the answer clear, complete and useful?

Return exactly this JSON format:
{{
  "retrieval_relevance": <int 1-5>,
  "answer_grounding": <int 1-5>,
  "answer_quality": <int 1-5>,
  "reasoning": "<one sentence explaining your scores>"
}}"""

    print("\nRunning self-evaluation...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0  # zero temperature = deterministic scoring
    )

    raw = response.choices[0].message.content.strip()

    # safely parse the JSON response
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # if LLM adds extra text, try to extract just the JSON part
        start = raw.find("{")
        end = raw.rfind("}") + 1
        scores = json.loads(raw[start:end])

    avg = round(
        (scores["retrieval_relevance"] +
         scores["answer_grounding"] +
         scores["answer_quality"]) / 3, 2
    )
    scores["average_score"] = avg

    print(f"  Retrieval relevance : {scores['retrieval_relevance']}/5")
    print(f"  Answer grounding    : {scores['answer_grounding']}/5")
    print(f"  Answer quality      : {scores['answer_quality']}/5")
    print(f"  Average             : {avg}/5")
    print(f"  Reasoning           : {scores['reasoning']}")

    return scores