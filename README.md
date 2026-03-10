# JPL Template Search — MCP Server

Connects Claude (via Claude Teams) to the JPL template library stored in Pinecone.
Deployed on Railway.

## What This Does

When anyone on the JPL team asks Claude about templates in any conversation,
Claude calls this server to search the Pinecone vector database and return
matching templates with their metadata.

## Tools Exposed

- **jpl_search_templates** — Semantic search by natural language query
- **jpl_get_template** — Retrieve full metadata for a specific template ID

## Environment Variables (set in Railway)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for generating query embeddings) |
| `PINECONE_API_KEY` | Pinecone API key (for searching the template index) |

## Deployment

1. Push this repo to GitHub
2. Connect the GitHub repo to Railway
3. Set environment variables in Railway dashboard
4. Railway auto-deploys on every push

## Registering with Claude Teams

Once deployed, register the Railway URL as a custom MCP connector in
Claude Teams organization settings. The URL will be:

    https://your-app-name.up.railway.app/mcp

## Local Testing

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export PINECONE_API_KEY="your-key"
python server.py
```
