# Google Agent Development Kit (ADK) - Implementation Guide

Implementation examples for agent design patterns using Google's Agent Development Kit (ADK).

> **Reference**: See [README.md](./README.md) for pattern descriptions, architectures, and use cases.

---

## Table of Contents

1. [Single-Agent System](#1-single-agent-system)
2. [Sequential Pipeline Pattern](#2-sequential-pipeline-pattern)
3. [Parallel Fan-Out/Gather Pattern](#3-parallel-fan-outgather-pattern)
4. [Coordinator/Dispatcher Pattern](#4-coordinatordispatcher-pattern)
5. [Hierarchical Task Decomposition Pattern](#5-hierarchical-task-decomposition-pattern)
6. [Loop Pattern](#6-loop-pattern)
7. [Generator & Critic Pattern](#7-generator--critic-pattern)
8. [Iterative Refinement Pattern](#8-iterative-refinement-pattern)
9. [Human-in-the-Loop Pattern](#9-human-in-the-loop-pattern)
10. [Memory Management Pattern](#10-memory-management-pattern)
11. [Tool Use Pattern](#11-tool-use-pattern)
12. [Guardrails/Safety Pattern](#12-guardrailssafety-pattern)

---

## 1. Single-Agent System

A single agent with an AI model, tools, and comprehensive system prompt handles the entire workflow autonomously.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Define tools
def search_database(query: str) -> str:
    """Search the database for information."""
    # Implementation here
    return f"Results for: {query}"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    # Implementation here
    return f"Email sent to {to}"

# Create single agent with tools
support_agent = LlmAgent(
    name="customer_support",
    model="gemini-2.0-flash",
    instruction="""You are a helpful customer support agent.
    Use the search_database tool to find relevant information.
    Use the send_email tool when customers need follow-up.""",
    tools=[
        FunctionTool(func=search_database),
        FunctionTool(func=send_email)
    ]
)
```

---

## 2. Sequential Pipeline Pattern

Agents execute in a predefined linear order where each agent's output feeds the next.

```python
from google.adk.agents import SequentialAgent, LlmAgent

# Define individual agents
parser_agent = LlmAgent(
    name="parser",
    model="gemini-2.0-flash",
    instruction="Parse the input and extract key information.",
    output_key="parsed_data"
)

extractor_agent = LlmAgent(
    name="extractor",
    model="gemini-2.0-flash",
    instruction="Extract entities from: {parsed_data}",
    output_key="extracted_data"
)

summarizer_agent = LlmAgent(
    name="summarizer",
    model="gemini-2.0-flash",
    instruction="Summarize the extracted data: {extracted_data}",
    output_key="summary"
)

# Create sequential pipeline
pipeline = SequentialAgent(
    name="data_pipeline",
    sub_agents=[parser_agent, extractor_agent, summarizer_agent]
)
```

---

## 3. Parallel Fan-Out/Gather Pattern

Multiple agents execute tasks simultaneously, then results are aggregated.

```python
from google.adk.agents import ParallelAgent, LlmAgent

# Define parallel review agents
security_agent = LlmAgent(
    name="security_auditor",
    model="gemini-2.0-flash",
    instruction="Review the code for security vulnerabilities.",
    output_key="security_review"  # Unique key to avoid race conditions
)

style_agent = LlmAgent(
    name="style_enforcer",
    model="gemini-2.0-flash",
    instruction="Check code style and formatting issues.",
    output_key="style_review"
)

perf_agent = LlmAgent(
    name="performance_analyst",
    model="gemini-2.0-flash",
    instruction="Analyze code for performance issues.",
    output_key="perf_review"
)

# Create parallel review system
review_system = ParallelAgent(
    name="code_review",
    sub_agents=[security_agent, style_agent, perf_agent]
)
# Note: Each agent writes to unique state key to avoid race conditions
```

---

## 4. Coordinator/Dispatcher Pattern

A central AI-powered agent analyzes requests and routes them to specialized sub-agents.

```python
from google.adk.agents import CoordinatorAgent, LlmAgent

# Define specialized agents
billing_agent = LlmAgent(
    name="billing_specialist",
    model="gemini-2.0-flash",
    description="Handles billing inquiries, payment issues, and invoices.",
    instruction="You are a billing specialist. Help with payment and invoice questions."
)

tech_support_agent = LlmAgent(
    name="tech_support",
    model="gemini-2.0-flash",
    description="Handles technical issues, troubleshooting, and product support.",
    instruction="You are a technical support specialist. Help diagnose and fix issues."
)

returns_agent = LlmAgent(
    name="returns_specialist",
    model="gemini-2.0-flash",
    description="Handles return requests, refunds, and exchanges.",
    instruction="You are a returns specialist. Process returns and refunds."
)

# Create coordinator - AutoFlow mechanism handles routing based on agent descriptions
support_system = CoordinatorAgent(
    name="customer_support",
    sub_agents=[billing_agent, tech_support_agent, returns_agent]
)
```

---

## 5. Hierarchical Task Decomposition Pattern

Multi-level hierarchy where high-level agents break down complex problems and delegate to specialized agents.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, FunctionTool

# Define sub-agents with their own tools
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

def summarize_text(text: str) -> str:
    """Summarize the given text."""
    return f"Summary: {text[:100]}..."

research_assistant = LlmAgent(
    name="research_assistant",
    model="gemini-2.0-flash",
    instruction="Research topics thoroughly using available tools.",
    tools=[
        FunctionTool(func=web_search),
        FunctionTool(func=summarize_text)
    ]
)

analysis_assistant = LlmAgent(
    name="analysis_assistant",
    model="gemini-2.0-flash",
    instruction="Analyze data and provide insights."
)

# Wrap sub-agents as tools for explicit calling by master agent
research_tool = AgentTool(agent=research_assistant)
analysis_tool = AgentTool(agent=analysis_assistant)

# Master agent that delegates to sub-agents
master_agent = LlmAgent(
    name="report_writer",
    model="gemini-2.0-flash",
    instruction="""You are a report writer. Break down complex reports into:
    1. Research phase - use research_assistant tool
    2. Analysis phase - use analysis_assistant tool
    3. Writing phase - synthesize results""",
    tools=[research_tool, analysis_tool]
)
```

---

## 6. Loop Pattern

Agents execute repeatedly until a termination condition is met.

```python
from google.adk.agents import LoopAgent, LlmAgent

# Define agents in the loop
check_agent = LlmAgent(
    name="status_checker",
    model="gemini-2.0-flash",
    instruction="""Check the current status.
    If the task is complete, set state['done'] = True.""",
    output_key="status"
)

process_agent = LlmAgent(
    name="processor",
    model="gemini-2.0-flash",
    instruction="Process based on the status: {status}",
    output_key="result"
)

# Create monitoring loop with max iterations to prevent infinite loops
monitor = LoopAgent(
    name="monitoring_loop",
    sub_agents=[check_agent, process_agent],
    max_iterations=10  # Safety limit
)
```

---

## 7. Generator & Critic Pattern

Separates content creation from validation with iterative improvement.

```python
from google.adk.agents import LoopAgent, SequentialAgent, LlmAgent

# Generator creates content
generator_agent = LlmAgent(
    name="generator",
    model="gemini-2.0-flash",
    instruction="""Generate code based on the requirements.
    Previous feedback: {critic_feedback}""",
    output_key="generated_code"
)

# Critic evaluates against criteria
critic_agent = LlmAgent(
    name="critic",
    model="gemini-2.0-flash",
    instruction="""Evaluate the code for:
    - Security vulnerabilities
    - Code style
    - Performance issues

    Code to review: {generated_code}

    If all criteria pass, respond with "APPROVED".
    Otherwise, provide specific feedback for improvement.""",
    output_key="critic_feedback"
)

# Combine into a review cycle
review_cycle = SequentialAgent(
    name="review_cycle",
    sub_agents=[generator_agent, critic_agent]
)

# Wrap in loop for iterative improvement
quality_loop = LoopAgent(
    name="quality_gate",
    sub_agents=[review_cycle],
    max_iterations=3  # Limit refinement cycles
)
```

---

## 8. Iterative Refinement Pattern

Focus on qualitative improvement over multiple cycles.

```python
from google.adk.agents import LoopAgent, LlmAgent

# Initial generator
generator = LlmAgent(
    name="draft_generator",
    model="gemini-2.0-flash",
    instruction="Generate initial draft or improve based on feedback: {refinement_notes}",
    output_key="draft"
)

# Critique agent provides optimization notes
critique = LlmAgent(
    name="critique_agent",
    model="gemini-2.0-flash",
    instruction="""Review the draft and provide specific improvement suggestions.
    Draft: {draft}

    If quality threshold is met, respond with "COMPLETE".
    Otherwise, provide actionable feedback.""",
    output_key="refinement_notes"
)

# Refiner polishes based on critique
refiner = LlmAgent(
    name="refiner",
    model="gemini-2.0-flash",
    instruction="""Polish and improve the draft based on feedback.
    Draft: {draft}
    Feedback: {refinement_notes}""",
    output_key="refined_output"
)

# Create refinement loop
refinement_loop = LoopAgent(
    name="refinement",
    sub_agents=[generator, critique, refiner],
    max_iterations=5
)
# Agent can signal escalate=True in EventActions for early exit
```

---

## 9. Human-in-the-Loop Pattern

Human authorization required for high-stakes, irreversible, or sensitive actions.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

def approval_tool(action_description: str) -> bool:
    """Pauses execution and requests human approval."""
    print(f"\n{'='*50}")
    print(f"APPROVAL REQUIRED")
    print(f"{'='*50}")
    print(f"Action: {action_description}")
    print(f"{'='*50}")
    response = input("Approve? (yes/no): ")
    return response.lower() == "yes"

def execute_transaction(amount: float, recipient: str) -> str:
    """Execute a financial transaction."""
    return f"Transaction of ${amount} to {recipient} completed."

# Agent with human approval gate
agent = LlmAgent(
    name="transaction_agent",
    model="gemini-2.0-flash",
    instruction="""You are a transaction agent.
    IMPORTANT: Before executing any transaction over $100,
    you MUST use the approval_tool to get human authorization.""",
    tools=[
        FunctionTool(func=approval_tool),
        FunctionTool(func=execute_transaction)
    ]
)
```

---

## 10. Memory Management Pattern

Maintains context and state across agent interactions.

```python
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent

# Session state for short-term memory (within conversation)
session_service = InMemorySessionService()

# Memory service for long-term storage (across conversations)
memory_service = InMemoryMemoryService()

# Create agent with memory capabilities
agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.0-flash",
    instruction="""You are a helpful assistant with memory.
    Use session state to track conversation context.
    Use memory service to recall information from past interactions."""
)

# Example: Creating a session
session = session_service.create_session(
    app_name="my_app",
    user_id="user_123"
)

# Store in session state
session.state["user_preferences"] = {"theme": "dark", "language": "en"}

# Store in long-term memory
memory_service.add_memory(
    app_name="my_app",
    user_id="user_123",
    memory_id="pref_001",
    content="User prefers dark theme and English language"
)
```

---

## 11. Tool Use Pattern

Agent uses external tools to extend capabilities beyond language generation.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
import requests

def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation with search API
    return f"Search results for: {query}"

def query_database(sql: str) -> str:
    """Execute a database query."""
    # Implementation with database connection
    return f"Query results for: {sql}"

def call_api(endpoint: str, method: str = "GET", data: dict = None) -> str:
    """Call an external API."""
    response = requests.request(method, endpoint, json=data)
    return response.text

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    # Safe evaluation implementation
    return eval(expression)  # Use proper safe_eval in production

# Agent with tool registry
agent = LlmAgent(
    name="tool_agent",
    model="gemini-2.0-flash",
    instruction="""You are an assistant with access to various tools.
    Use the appropriate tool based on the user's request:
    - search_web: For finding information online
    - query_database: For data retrieval
    - call_api: For external service integration
    - calculate: For mathematical computations""",
    tools=[
        FunctionTool(func=search_web),
        FunctionTool(func=query_database),
        FunctionTool(func=call_api),
        FunctionTool(func=calculate)
    ]
)
```

---

## 12. Guardrails/Safety Pattern

Validates inputs and outputs to ensure safety, compliance, and quality.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import FunctionTool

def check_toxicity(text: str) -> dict:
    """Check text for toxic content."""
    # Implementation with content moderation API
    return {"is_toxic": False, "score": 0.1}

def detect_pii(text: str) -> dict:
    """Detect personally identifiable information."""
    # Implementation with PII detection
    return {"has_pii": False, "entities": []}

def check_injection(text: str) -> dict:
    """Check for prompt injection attempts."""
    # Implementation with injection detection
    return {"is_injection": False, "confidence": 0.95}

# Input guardrails agent
input_guard = LlmAgent(
    name="input_guardrails",
    model="gemini-2.0-flash",
    instruction="""Validate the input using safety tools.
    If any check fails, respond with "BLOCKED: [reason]".
    Otherwise, pass the input through unchanged.""",
    tools=[
        FunctionTool(func=check_toxicity),
        FunctionTool(func=detect_pii),
        FunctionTool(func=check_injection)
    ],
    output_key="validated_input"
)

# Main processing agent
main_agent = LlmAgent(
    name="main_processor",
    model="gemini-2.0-flash",
    instruction="Process the validated input: {validated_input}",
    output_key="raw_output"
)

# Output guardrails agent
output_guard = LlmAgent(
    name="output_guardrails",
    model="gemini-2.0-flash",
    instruction="""Validate the output for:
    - Factual accuracy
    - Safety compliance
    - No PII leakage

    Output to validate: {raw_output}

    If any check fails, respond with a safe alternative.
    Otherwise, pass the output through.""",
    output_key="final_output"
)

# Create guarded pipeline
guarded_agent = SequentialAgent(
    name="guarded_system",
    sub_agents=[input_guard, main_agent, output_guard]
)
```

---

## Running Agents

Basic example of running an ADK agent:

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Initialize services
session_service = InMemorySessionService()
runner = Runner(
    agent=your_agent,
    app_name="my_app",
    session_service=session_service
)

# Create a session
session = session_service.create_session(
    app_name="my_app",
    user_id="user_123"
)

# Run the agent
async def main():
    response = await runner.run(
        session_id=session.id,
        user_id="user_123",
        message="Hello, how can you help me?"
    )
    print(response)

import asyncio
asyncio.run(main())
```

---

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Developer's guide to multi-agent patterns in ADK](https://developers.googleblog.com/en/developers-guide-to-multi-agent-patterns-in-adk/)
- [ADK GitHub Repository](https://github.com/google/adk-python)
