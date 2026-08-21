from openai import OpenAI


MODEL_NAME = "qwen2.5:3b"


class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def generate_answer(
        self,
        query: str,
        search_results: list[dict],
    ) -> str:
        context_parts = []

        for result in search_results:
            source = (
                f"{result['path']}:"
                f"{result['start_line']}-{result['end_line']}"
            )

            context_parts.append(
                f"SOURCE: {source}\n"
                f"SYMBOL: {result.get('symbol')}\n"
                f"CODE:\n{result['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are RepoAgent, a concise codebase assistant.

Your job is to help developers understand a repository faster than using a
general-purpose LLM.

Answer the user's question using ONLY the provided repository context.

Rules:
- Keep the answer short and practical.
- Prefer 2-5 sentences.
- Use bullets only when they make the answer easier to scan.
- Do not give long introductions or conclusions.
- Do not repeat the user's question.
- Do not explain obvious programming concepts unless needed.
- Mention the exact function, class, or file responsible.
- Only state implementation details explicitly supported by the provided code.
- Do not guess or invent behavior.
- Cite relevant claims using [path:start-end].
- Usually include no more than 2 citations.
- If the context is insufficient, say so briefly.
- Focus on "where", "how", and "what happens next".
- Silently interpret obvious typos using the repository context.
- Never comment on spelling mistakes unless they change the meaning materially.
- Answer the likely intended technical question directly.
- Prefer 1-3 sentences.
- Start with the exact function/class/file responsible.
- Avoid filler such as "RepoAgent finds..." when a direct answer is clearer.
- Do not invent examples, filenames, extensions, or behaviors that are not explicitly shown in the retrieved code.
- If explaining a filter or condition, describe the rule rather than adding your own examples.

USER QUESTION:
{query}

REPOSITORY CONTEXT:
{context}
"""

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()