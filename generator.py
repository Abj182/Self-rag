from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY 
the context provided. If the context doesn't contain enough information, 
say "This doesn't appear in the indexed documents."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # free, very fast
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    return answer