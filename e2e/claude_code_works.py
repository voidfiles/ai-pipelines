import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)

from ai_pipelines.sdk_patch import apply as apply_sdk_patch

apply_sdk_patch()

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

async def main():
    async for message in query(
        prompt="Say hi. Then find all the files in @e2e/ and tell me what they are.",
        options=ClaudeAgentOptions(
            stderr=lambda line: print(f"STDERR: {line}"),
        ),
    ):
        if message is None:
            continue

        print(type(message).__name__, message)

        if isinstance(message, ResultMessage):
            print(f"Result: {message.result}")
            print(f"Cost: ${message.total_cost_usd}")

            if message.is_error:
                print(f"ERROR: {message.result}")


asyncio.run(main())
