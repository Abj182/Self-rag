from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate(query, context_chunks):
    if not context_chunks:
        raise ValueError("No context chunks provided to generator.")

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY 
the context provided. If the context doesn't contain enough information, 
say "This doesn't appear in the indexed documents."

Context:
{context}

Question: {query}

Answer:"""

    print(f"\nCalling LLM for query: '{query[:60]}...'")

    try:
        response = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
            timeout     = 30
        )
        answer = response.choices[0].message.content.strip()

        if not answer:
            raise ValueError("LLM returned an empty response.")

        print(f"LLM answer: {answer[:100]}...")
        return answer

    except Exception as e:
        print(f"Generator error: {e}")
        raise