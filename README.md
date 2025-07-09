# ai-evals-context-talk

Companion repo to my talk for Hamel and Shreya's AI Evals course about AI assisted coding for eval tools.

## Structure

This show examples based on the lecture approach for 3 eval tools, you can see the directories for them:

- braintrust
- inspect
- pheonix

Inside each directory there is:

- General Context
- Curated Context
- Personalized Context

# Talk 

## Optimize Your Dev Setup For Evals w/ Cursor Rules & MCP

### Why bother?

- Better AI asistance
- Training cutoff may mean outdated info
- Spending time determing what's important to your project is good

### The Three layers

1. General Context: Generic tool that works on almost anything
2. Curated Context: Curated by an expert, such as the tool author
3. Personalized Context:  Context that you can create that's unique to your project

### General Context

- Git MCP
    - https://gitmcp.io/UKGovernmentBEIS/inspect_evals/chat
    - 
- Repo Mix
    - [Braintrust]
* Note prompt injection risk

### Curated Context

- llms.txt
    - https://github.com/langchain-ai/mcpdoc
    - https://inspect.aisi.org.uk/llms.txt
    - https://inspect.aisi.org.uk/llms-full.txt
- jina AI reader
- Manual copy paste

### Personalized Context

- Filtering uneeded content
- Replacing examples
- Choosing best practices

### Cursor Rules

- Always Apply
- Apply Manually
- Apply Intelligently
- Apply to Specific Files
