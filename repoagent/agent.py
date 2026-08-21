import json

from openai import OpenAI

from repoagent.tools import RepoTools


MODEL_NAME = "qwen2.5:3b"


class RepoAgent:
    def __init__(self, tools: RepoTools):
        self.tools = tools

        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": (
                        "Search repository code for concepts, behaviors, "
                        "features, or implementation details."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Description of the implementation "
                                    "or behavior to search for."
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_symbol",
                    "description": (
                        "Find an exact function or class by symbol name. "
                        "Use only when the exact symbol is known."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                            }
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a known repository file when more surrounding "
                        "context is needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": (
                        "List source files in the repository. "
                        "Use only for repository structure questions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]

    def run(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are RepoAgent, a concise codebase investigation assistant. "
                    "Use repository tools before answering implementation questions. "
                    "Use search_code for concepts, behaviors, features, and "
                    "implementation questions. "
                    "For HOW or WHERE questions, search_code should usually be "
                    "the first tool you use. "
                    "Use find_symbol only when the user provides an exact function "
                    "or class name. "
                    "Use read_file only after you already know which file is relevant "
                    "and need more surrounding context. "
                    "Use list_files only for questions about project structure "
                    "or which files exist. "
                    "Prefer core source code over examples, tests, and docs. "
                    "Never invent implementation details, APIs, examples, files, "
                    "or behavior that were not returned by repository tools. "
                    "Final answers should normally be 1-3 sentences. "
                    "Answer the exact repository question directly. "
                    "Mention the most relevant file and symbol when known."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        max_steps = 6

        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto",
                temperature=0,
            )

            message = response.choices[0].message

            if not message.tool_calls:
                if message.content:
                    return message.content.strip()

                return "I could not produce an answer."

            messages.append(
                message.model_dump(exclude_none=True)
            )

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )
                except json.JSONDecodeError:
                    arguments = {}

                tool_result = self._execute_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        return "I could not answer within the tool-call limit."

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> str:
        if tool_name == "search_code":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)

            try:
                top_k = int(top_k)
            except (TypeError, ValueError):
                top_k = 5

            results = self.tools.search_code(
                query=query,
                top_k=top_k,
            )

            simplified = []

            for result in results:
                simplified.append(
                    {
                        "source": (
                            f"{result['path']}:"
                            f"{result['start_line']}-"
                            f"{result['end_line']}"
                        ),
                        "symbol": result.get("symbol"),
                        "type": result.get("type"),
                        "content": result["content"][:2500],
                    }
                )

            return json.dumps(
                simplified,
                indent=2,
            )

        if tool_name == "find_symbol":
            symbol = arguments.get("symbol", "")

            results = self.tools.find_symbol(symbol)

            simplified = []

            for result in results:
                simplified.append(
                    {
                        "source": (
                            f"{result['path']}:"
                            f"{result['start_line']}-"
                            f"{result['end_line']}"
                        ),
                        "symbol": result.get("symbol"),
                        "type": result.get("type"),
                        "content": result["content"][:2500],
                    }
                )

            return json.dumps(
                simplified,
                indent=2,
            )

        if tool_name == "read_file":
            path = arguments.get("path", "")

            return self.tools.read_file(path)[:8000]

        if tool_name == "list_files":
            return json.dumps(
                self.tools.list_files(),
                indent=2,
            )

        return f"Unknown tool: {tool_name}"