"""
Verify that qwen3:30b can produce a structured upsert_node tool call.
"""
import json
import sys

import ollama

UPSERT_NODE_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_node",
        "description": (
            "Insert or update a node in the knowledge graph. "
            "Call this for every entity mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical entity name",
                },
                "label": {
                    "type": "string",
                    "description": "Entity type, e.g. Person, Place, Concept",
                },
                "props": {
                    "type": "object",
                    "description": "Arbitrary key-value metadata about the entity",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name", "label"],
        },
    },
}


def main():
    client = ollama.Client(host="http://localhost:11434")

    prompt = (
        "Extract every entity from this sentence and call upsert_node for each one:\n"
        '"Marie Curie was a physicist who worked in Paris."'
    )

    print("Sending request to qwen3:30b …")
    response = client.chat(
        model="qwen3:30b",
        messages=[{"role": "user", "content": prompt}],
        tools=[UPSERT_NODE_TOOL],
    )

    msg = response.message
    calls = msg.tool_calls or []

    if not calls:
        print("FAIL — no tool calls returned.")
        print("Model text response:", msg.content)
        sys.exit(1)

    print(f"PASS — {len(calls)} tool call(s) received:\n")
    for tc in calls:
        fn = tc.function.name
        args = tc.function.arguments  # dict, parsed by ollama client
        print(f"  {fn}({json.dumps(args, indent=4)})")


if __name__ == "__main__":
    main()
