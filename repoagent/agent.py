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
                        "Semantically search the repository for code related "
                        "to a concept, behavior, or implementation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What code to search for.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": (
                                    "Number of search results to return."
                                ),
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
                        "Find an exact function or class by its symbol name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": (
                                    "Exact function or class name."
                                ),
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
                        "Read the complete contents of a known repository file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "File path relative to the repository root."
                                ),
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
                        "List source files in the repository. Use this for "
                        "questions about repository structure, not code behavior."
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
                    "You are RepoAgent, a concise codebase assistant. "
                    "Use repository tools to investigate questions before "
                    "answering. For questions about how or where behavior is "
                    "implemented, prefer search_code. If the user explicitly "
                    "names a function or class, prefer find_symbol. "
                    "Use read_file when you need more surrounding context. "
                    "Use list_files only for repository structure questions. "
                    "Never invent implementation details. "
                    "Keep final answers to 1-4 short sentences and mention "
                    "relevant functions/files."
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

            # If there are no tool calls, the agent is done.
            if not message.tool_calls:
                if message.content:
                    return message.content.strip()

                return "I could not produce an answer."

            # Preserve the assistant tool-call message.
            messages.append(
                message.model_dump(
                    exclude_none=True,
                )
            )

            # Execute every requested tool.
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
                        "content": result["content"],
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
                        "content": result["content"],
                    }
                )

            return json.dumps(
                simplified,
                indent=2,
            )

        if tool_name == "read_file":
            path = arguments.get("path", "")

            return self.tools.read_file(path)

        if tool_name == "list_files":
            return json.dumps(
                self.tools.list_files(),
                indent=2,
            )

        return f"Unknown tool: {tool_name}"