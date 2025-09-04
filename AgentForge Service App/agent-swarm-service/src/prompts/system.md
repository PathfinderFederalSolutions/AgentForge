# System Prompts and Guidelines for Agent Swarm

## Overview
This document outlines the system prompts and guidelines for the agents within the agent swarm system. It serves as a reference for expected behaviors, output formats, and interaction protocols among agents.

## Agent Behavior
1. **Data Processing**: Agents must efficiently process incoming data, applying relevant algorithms and logic to derive insights.
2. **Communication**: Agents should communicate with each other to share data and results, ensuring a smooth workflow.
3. **Error Handling**: Agents must implement robust error handling to manage unexpected situations gracefully.

## Output Formats
- All agents should return results in a standardized format, which includes:
  - **Status**: Indicating success or failure of the operation.
  - **Data**: The processed data or insights derived from the input.
  - **Error Messages**: Clear and concise error messages when applicable.

## Interaction Protocols
- Agents must adhere to the following protocols when interacting:
  - **Request-Response Model**: Agents should send requests to each other and wait for responses before proceeding.
  - **Timeouts**: Implement timeouts for requests to avoid indefinite waiting periods.
  - **Logging**: All interactions should be logged for auditing and debugging purposes.

## User Prompts
- Agents should be capable of interpreting user prompts and transforming them into actionable tasks. The expected workflow is as follows:
  1. Receive user prompt.
  2. Parse the prompt to identify required actions.
  3. Delegate tasks to appropriate agents.
  4. Compile results and deliver them back to the user.

## Conclusion
These guidelines are essential for maintaining a cohesive and efficient agent swarm system. Adherence to these prompts will ensure that agents operate effectively and deliver high-quality results based on user inputs.