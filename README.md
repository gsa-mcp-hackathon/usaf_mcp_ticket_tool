# AndOff Aircraft Maintenance Web App & MCP Server

A unified **Flask + FastAPI + FastMCP** application built for the GSA Hackathon to explore **Model Context Protocol (MCP)** server functionality on Databricks Apps.

It pairs a dark-mode **AndOff Issue Tracking Dashboard & AI Assistant** (`/`) with a **Streamable HTTP MCP Server** (`/mcp`) so AI agents can query, search, and trigger alerts on Unity Catalog maintenance data directly.

---

## Architecture

- **Unified ASGI Entrypoint (`FastAPI`)**: Hosts both the web UI and the MCP server on port `8000`.
  - `/` — Flask Web Application (`WSGIMiddleware`) for viewing/creating AndOff issues and chatting with the AI Maintenance Assistant.
  - `/mcp` — `FastMCP` (`andoff-mcp-server`) exposing Unity Catalog tools over Streamable HTTP.
- **Data Layer**: Connects to a Databricks SQL Warehouse (`use_cloud_fetch=False` for private/government networks) and queries Unity Catalog tables:
  - `workspace.default.databricks_issue_import_filtered`
  - `workspace.default.t_org_list`
  - `workspace.default.t_user_accounts`
- **AI Assistant (`/chat`)**: Supports Databricks Model Serving & AI Gateway models (`gpt-5-6-terra`, `gpt-5-6-luna`, `grok-4-6`, etc.) with live Unity Catalog issue context and direct Tier 1 alert triggering.

---

## Exposed MCP Tools (`/mcp`)

AI agents connecting to `/mcp` can invoke the following tools:

| Tool | Description |
|:---|:---|
| `query_catalog_table(query)` | Run a read-only SQL query against Unity Catalog tables. |
| `get_andoff_issues(limit=20)` | Retrieve AndOff maintenance issues ordered by `PriorityRank`, including `Status`. |
| `search_andoff_issues(keyword)` | Search issues by keyword across `IssueDesc`, `IssueTypeId`, `SubIssueType`, `PartNum`, or `Status`. |
| `get_organizations()` | List all organizations from `t_org_list`. |
| `get_user_accounts()` | List active user accounts and roles from `t_user_accounts`. |
| `send_tier1_sql_alert(job_id="")` | Trigger the Databricks Job (`DATABRICKS_JOB_ID`) to evaluate the Tier 1 SQL Alert and send email notifications. |

---

## Project Structure

```text
src/app/
├── .env.example        # Safe template for required environment variables
├── .gitignore          # Excludes .env and Python cache files from Git
├── app.py              # Unified FastAPI + Flask + FastMCP application
├── requirements.txt    # Python dependencies
├── static/             # Static assets
└── templates/          # Flask HTML templates (index.html)

