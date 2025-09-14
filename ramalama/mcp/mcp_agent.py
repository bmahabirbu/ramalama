#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import urllib.request
import openai
from typing import Any, Dict, List, Optional, Union

from fastmcp import Client
from fastmcp.client.auth import OAuth

class LLMAgent:
    """An LLM-powered agent that can use MCP tools to accomplish tasks."""

    def __init__(self, client_urls: List[str], llm_base_url: str = "http://localhost:8080"):
        self.client_urls = client_urls
        self.llm_base_url = llm_base_url.rstrip("/")
        self.client_tools: Dict[Client, List[Tool]] = {}
        self._stream_callback = None
        self.llm = None
        # initialize LLM
        self.initialize_llm()

    async def initialize_mcp(self):
        """Connect to MCP servers and fetch tools."""
        for url in self.client_urls:
            try:
                mcp_client = Client(url, auth=oauth)
                async with mcp_client as connected_client:
                    await connected_client.ping()
                    tools = await connected_client.list_tools()
                    self.client_tools[connected_client] = tools
            except Exception as e:
                logging.error(f"Failed to connect to {url}: {e}")

        if not self.client_tools:
            raise RuntimeError("No tools available from any server")
    
    def initialize_llm(self):
        """Initialize the LLM."""
        self.llm = openai.OpenAI(api_key="your-api-key", base_url=f"{self.llm_base_url}")

    def print_tools(self):
        """Print all tools with numbering per client."""
        for client, tools in self.client_tools.items():
            print(f"\nTools for client: {client.name}\n{'-'*30}")
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool.name} - {tool.description}")

    
    def get_available_tools(self) -> List[str]:
        """Return a flat list of all tool names, regardless of client."""
        tool_names = []
        for tools in self.client_tools.values():
            for tool in tools:
                tool_names.append(tool)
        return tool_names
    
    async def execute_specific_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a specific tool by name with given arguments."""

        # find the client that has the tool
        client = next(
            (c for c, tools in self.client_tools.items() if any(t.name == tool_name for t in tools)),
            None
        )
        if not client:
            return f"Tool '{tool_name}' not found."

        # execute the tool with the corresponding client
        async with client as connected_client:
            result = await connected_client.call_tool(tool_name, arguments)
            return result.data
    
    def should_use_tools(self, prompt: str, conversation_history: list = None) -> bool:
        """Determine if the request should be handled by tools using LLM."""
        
        # get all tools
        all_tools = self.get_available_tools()
        
        # Build the tools context
        tools_context = "Available tools:\n"
        for i, tool in enumerate(all_tools, 1):
            tool_name = getattr(tool, "name", "<no name>")
            tool_desc = getattr(tool, "description", "")
            tools_context += f"{i}. {tool_name}: {tool_desc}\n"

        # build the conversation history context
        context_info = ""
        if conversation_history:
            context_info = "\nRecent conversation:\n"
            # limit the conversation history to the last 3 messages
            for msg in conversation_history[-3:]:
                role = msg['role'].capitalize()
                preview = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
                context_info += f"{role}: {preview}\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent assistant that determines whether a user's "
                    "request should be handled by available tools or by regular conversation.\n\n"
                    "Answer ONLY 'YES' if tools should be used or 'NO' otherwise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request: {prompt}\n"
                    f"{context_info}\n"
                    f"{tools_context}\n"
                    "Should this request use the available tools?"
                ),
            },
        ]

        response = self.llm.chat.completions.create(
            model="your-model-name", messages=messages, stream=False
        )
        content = response.choices[0].message.content if response.choices else ""
        return content.upper().strip() == "YES"

    async def execute_task(self, task: str, stream: bool = False) -> str:
        """Select tools with LLM and run them."""

        # Flatten all tools
        all_tools = [tool for tools in self.client_tools.values() for tool in tools]

        # Build tool list context for LLM selection
        tool_list_text = "\n".join(f"- {t.name}: {t.description}" for t in all_tools)

        # Ask LLM which tools to use
        response = self.llm.chat.completions.create(
            model="your-model-name",
            messages=[
                {"role": "system", "content": "Select all relevant tools as a comma-separated list, or NONE."},
                {"role": "user", "content": f"Task: {task}\n\nAvailable tools:\n{tool_list_text}"}
            ],
            stream=False,
        )

        content = response.choices[0].message.content if response.choices else ""
        chosen_names = [name.strip().lower() for name in content.split(",")] if content and content.upper() != "NONE" else []

        # Find the actual tool objects
        selected_tools = [t for t in all_tools if t.name.lower() in chosen_names]
        if not selected_tools:
            return "No relevant tools selected."

        results = []
        for tool in selected_tools:
            # Find the client that owns this tool
            client = next(c for c, tools in self.client_tools.items() if tool in tools)
            try:
                arg_prompt = [
                    {"role": "system", "content": "Generate JSON arguments for this tool."},
                    {"role": "user", "content": f"Tool: {tool.name}\nTask: {task}\nSchema: {tool.inputSchema}"}
                ]
                arg_resp = self.llm.chat.completions.create(
                    model="your-model-name",
                    messages=arg_prompt,
                    stream=False
                )
                args_text = arg_resp.choices[0].message.content if arg_resp.choices else "{}"
                try:
                    args = json.loads(args_text)
                except json.JSONDecodeError:
                    args = {}

                # Execute tool
                async with client as connected_client:
                    result = await connected_client.call_tool(tool.name, args)
                    results.append(f"{tool.name} result:\n{getattr(result, 'data', str(result))}")

            except Exception as e:
                results.append(f"{tool.name} error: {e}")
                logging.error(f"Tool execution error: {tool.name}", exc_info=True)

        # Summarize results using LLM
        summary_prompt = [
            {"role": "system", "content": "Summarize results into a helpful answer."},
            {"role": "user", "content": f"Task: {task}\n\nRaw results:\n{json.dumps(results, indent=2)}"}
        ]
        
        if stream and callable(self._stream_callback):
            # Handle streaming response
            summary_resp = self.llm.chat.completions.create(
                model="your-model-name",
                messages=summary_prompt,
                stream=True
            )
            
            for chunk in summary_resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    self._stream_callback(chunk.choices[0].delta.content)
            
            print()
            return ""  
        else:
            # Handle non-streaming response
            summary_resp = self.llm.chat.completions.create(
                model="your-model-name",
                messages=summary_prompt,
                stream=False
            )
            return summary_resp.choices[0].message.content if summary_resp.choices else "\n".join(results)

    # Sync wrappers
    def initialize_sync(self):
        return asyncio.run(self.initialize_mcp())

    def execute_task_sync(self, task: str, stream: bool = False) -> str:
        return asyncio.run(self.execute_task(task, stream))

    def execute_specific_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        return asyncio.run(self.execute_specific_tool(tool_name, arguments))
