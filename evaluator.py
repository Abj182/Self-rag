from groq import Groq
import os
import json

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluate(query, chunks, answer):
    if not chunks or not answer:
        raise ValueError("Cannot evaluate — missing chunks or answer.")

    context = "\n\n---\n\n".join(chunks[:3])  # limit context to save tokens

    eval_prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the following and return ONLY a JSON object, no extra text, no markdown.

Query: {query}

Retrieved Context:
{context}

Generated Answer: {answer}

Score each dimension from 1 to 5:
- retrieval_relevance: Are the retrieved chunks relevant to the query?
- answer_grounding: Is the answer fully supported by the context?
- answer_quality: Is the answer clear, complete and useful?

Return ONLY this JSON, nothing else:
{{"retrieval_relevance": <int>, "answer_grounding": <int>, "answer_quality": <int>, "reasoning": "<one sentence>"}}"""

    print("\nRunning self-evaluation...")

    try:
        response = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = [{"role": "user", "content": eval_prompt}],
            temperature = 0.0,
            timeout     = 30
        )

        raw = response.choices[0].message.content.strip()

        # strip markdown code fences if LLM adds them
        raw = raw.replace("```json", "").replace("```", "").strip()

        # extract JSON block if there's surrounding text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in evaluator response: {raw}")

        scores = json.loads(raw[start:end])

        # validate all required keys exist and are ints 1-5
        for key in ["retrieval_relevance", "answer_grounding", "answer_quality"]:
            if key not in scores:
                scores[key] = 3  # default to neutral if missing
            scores[key] = max(1, min(5, int(scores[key])))

        if "reasoning" not in scores:
            scores["reasoning"] = "No reasoning provided."

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

    except json.JSONDecodeError as e:
        print(f"Evaluator JSON parse error: {e} — raw: {raw}")
        raise ValueError(f"Evaluator returned invalid JSON: {e}")
    except Exception as e:
        print(f"Evaluator error: {e}")
        raise