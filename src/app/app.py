import os
import json
import uuid
import tempfile
import urllib.request
import urllib.error
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session

# FastAPI + WSGI bridge + FastMCP for hosting Web App (/) and MCP Server (/mcp) together
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from mcp.server.fastmcp import FastMCP

# Bypasses the Spark namespace collision by using the explicit package path
from databricks.sql import connect as databricks_connect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

app = Flask(__name__)

# Load .env from the same directory as app.py (no external packages required)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# --- Context Window & Table Configuration ---
MAX_HISTORY_MESSAGES = 100
MAX_DB_CONTEXT_ROWS = 50
MAX_OUTPUT_TOKENS = 4096

ISSUES_TABLE = "workspace.default.databricks_issue_import_filtered"
ORGS_TABLE = "workspace.default.t_org_list"
USERS_TABLE = "workspace.default.t_user_accounts"
CHAT_STORE_FILE = os.path.join(tempfile.gettempdir(), "andoff_chat_sessions.json")


def load_chat_history(session_id):
    try:
        if os.path.exists(CHAT_STORE_FILE):
            with open(CHAT_STORE_FILE, "r", encoding="utf-8") as f:
                store = json.load(f)
                return store.get(session_id, [])
    except Exception:
        pass
    return []


def save_chat_history(session_id, history):
    try:
        store = {}
        if os.path.exists(CHAT_STORE_FILE):
            try:
                with open(CHAT_STORE_FILE, "r", encoding="utf-8") as f:
                    store = json.load(f)
            except Exception:
                store = {}
        store[session_id] = history[-MAX_HISTORY_MESSAGES:]
        if len(store) > 200:
            oldest_key = next(iter(store))
            store.pop(oldest_key, None)
        with open(CHAT_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except Exception:
        pass


# --- Available Agent Models ---
AVAILABLE_MODELS = [
    {"id": "gpt-5-6-terra", "label": "GPT-5.6 Terra (databricks-gpt-5-6-terra)"},
    {"id": "gpt-5-6-luna", "label": "GPT-5.6 Luna (databricks-gpt-5-6-luna)"},
    {"id": "grok-4-6", "label": "Grok 4.6 (databricks-grok-4-6)"},
    {"id": "gemini-3-8-flash", "label": "Gemini 3.8 Flash (databricks-gemini-3-8-flash)"},
    {"id": "gemini-3-7-flash", "label": "Gemini 3.7 Flash (databricks-gemini-3-7-flash)"},
    {"id": "gemini-3-6-flash", "label": "Gemini 3.6 Flash (databricks-gemini-3-6-flash)"},
    {"id": "gemini-3-5-flash", "label": "Gemini 3.5 Flash (databricks-gemini-3-5-flash)"},
    {"id": "gemini-3-5-flash-lite", "label": "Gemini 3.5 Flash Lite (databricks-gemini-3-5-flash-lite)"},
    {"id": "gpt-5-4", "label": "GPT-5.4 (databricks-gpt-5-4)"}
]
VALID_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}


def get_server_hostname():
    host = os.environ.get("DATABRICKS_HOST")
    if not host:
        raise RuntimeError(
            "DATABRICKS_HOST is not set. Add it to the .env file in the same folder as app.py."
        )
    return host.replace("https://", "").strip("/")


def get_pat_token():
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError(
            "DATABRICKS_TOKEN is not set. Add it to the .env file in the same folder as app.py."
        )
    return token


def get_warehouse_id():
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not set. Add it to the .env file in the same folder as app.py."
        )
    return warehouse_id


def get_logged_in_user_email():
    """Returns the authenticated user's email from Databricks Apps reverse-proxy headers."""
    return (
        request.headers.get("X-Forwarded-Email")
        or request.headers.get("X-Forwarded-User")
        or "local-dev@us.af.mil"
    )


# --- Unified SQL Warehouse Connection Function ---
def get_db_connection():
    return databricks_connect(
        server_hostname=get_server_hostname(),
        http_path=f"/sql/1.0/warehouses/{get_warehouse_id()}",
        access_token=get_pat_token(),
        use_cloud_fetch=False  # Crucial fix for private government networks
    )


def get_andoff_context(limit=MAX_DB_CONTEXT_ROWS):
    """Fetches recent AndOff issues from Unity Catalog for the AI Assistant, including Status."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT IssueId, OrgId, IssueTypeId, SubIssueType, PriorityRank, IssueDesc, PartNum, Status
                    FROM {ISSUES_TABLE}
                    ORDER BY PriorityRank ASC, IssueId DESC
                    LIMIT {int(limit)}
                """)
                rows = cursor.fetchall()
                if not rows:
                    return "No AndOff issues found."
                lines = []
                for r in rows:
                    lines.append(
                        f"- {r[0]} | Org: {r[1]} | Type: {r[2]} ({r[3]}) | Priority: {r[4]} | Status: {r[7]} | Part: {r[6]} | Desc: {r[5]}"
                    )
                return "\n".join(lines)
    except Exception as e:
        return f"Database context unavailable: {e}"


def ask_databricks_agent(prompt_text, model_id="gpt-5-6-terra", chat_history=None):
    if model_id not in VALID_MODEL_IDS:
        model_id = "gpt-5-6-terra"
    server_hostname = get_server_hostname()
    pat_token = get_pat_token()
    unity_gateway_model = f"system.ai.{model_id}"
    serving_endpoint_name = f"databricks-{model_id}"

    db_context = get_andoff_context(limit=MAX_DB_CONTEXT_ROWS)
    system_instruction = (
        "You are an USAF AndOff Aircraft Maintenance AI Assistant running on Databricks. "
        "You help maintenance teams analyze open aircraft issues, prioritize blockers (PriorityRank 1 = highest), "
        "track issue status, and recommend troubleshooting or workarounds.\n\n"
        f"Below are the current AndOff issues from {ISSUES_TABLE}:\n"
        f"{db_context}\n\n"
        "Answer the user's prompt clearly and concisely."
    )

    sdk_messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_instruction)]
    ug_input = [
        {"role": "user", "content": [{"type": "input_text", "text": system_instruction}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Understood. How can I assist with AndOff maintenance issues today?"}]}
    ]

    if chat_history and isinstance(chat_history, list):
        for turn in chat_history[-MAX_HISTORY_MESSAGES:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if not content:
                continue
            if role == "user":
                sdk_messages.append(ChatMessage(role=ChatMessageRole.USER, content=content))
                ug_input.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
            elif role == "assistant":
                sdk_messages.append(ChatMessage(role=ChatMessageRole.ASSISTANT, content=content))
                ug_input.append({"role": "assistant", "content": [{"type": "output_text", "text": content}]})

    sdk_messages.append(ChatMessage(role=ChatMessageRole.USER, content=prompt_text))
    ug_input.append({"role": "user", "content": [{"type": "input_text", "text": prompt_text}]})

    try:
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=serving_endpoint_name,
            messages=sdk_messages,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        return response.choices[0].message.content
    except Exception as sp_err:
        sp_error_msg = str(sp_err)

    ug_url = f"https://{server_hostname}/ai-gateway/mlflow/v1/responses"
    ug_payload = {
        "model": unity_gateway_model,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": ug_input
    }
    req = urllib.request.Request(
        ug_url,
        data=json.dumps(ug_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {pat_token}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "output_text" in data and data["output_text"]:
                return data["output_text"]
            if "output" in data and isinstance(data["output"], list):
                texts = [
                    block["text"]
                    for item in data["output"]
                    for block in item.get("content", [])
                    if block.get("type") in ("output_text", "text") and "text" in block
                ]
                if texts:
                    return "\n".join(texts)
    except Exception as ug_err:
        return f"Agent Error ({serving_endpoint_name}): Container SP Error: {sp_error_msg} | PAT Fallback Error: {ug_err}"


# =====================================================================
# --- Flask Routes (AndOff Web Application UI & APIs) ---
# =====================================================================

@app.route('/')
def home():
    """Renders the dark-mode AndOff Dashboard & 5-Step Wizard UI with Status included."""
    issues, orgs, users = [], [], []
    db_status = "Connected"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT IssueId, OrgId, IssueTypeId, SubIssueType, PriorityRank, IssueDesc, PartNum, IssuePoc, IssuePocEmail, Status
                    FROM {ISSUES_TABLE}
                    ORDER BY PriorityRank ASC, IssueId DESC
                    LIMIT 100
                """)
                issues = [
                    {
                        "IssueId": r[0], "OrgId": r[1], "IssueTypeId": r[2],
                        "SubIssueType": r[3], "PriorityRank": r[4], "IssueDesc": r[5],
                        "PartNum": r[6], "IssuePoc": r[7], "IssuePocEmail": r[8],
                        "Status": r[9] if r[9] is not None else "Open"
                    }
                    for r in cursor.fetchall()
                ]
                cursor.execute(f"SELECT OrgId, Org, OrgDesc FROM {ORGS_TABLE} ORDER BY OrgId ASC")
                orgs = [{"OrgId": r[0], "Org": r[1], "OrgDesc": r[2]} for r in cursor.fetchall()]
                cursor.execute(f"""
                    SELECT UserAccountId, DisplayName, Email, Role, OrgId, CreatedDate, IsActive
                    FROM {USERS_TABLE}
                    WHERE IsActive = 1
                    ORDER BY DisplayName ASC
                """)
                users = [
                    {
                        "UserAccountId": r[0], "DisplayName": r[1], "Email": r[2],
                        "Role": r[3], "OrgId": r[4], "CreatedDate": str(r[5]), "IsActive": r[6]
                    }
                    for r in cursor.fetchall()
                ]
    except Exception as e:
        db_status = f"Database Error: {e}"

    user_email = get_logged_in_user_email()
    current_user = next(
        (u for u in users if u["Email"] and u["Email"].lower() == user_email.lower()),
        {"DisplayName": user_email, "Email": user_email, "Role": "User", "OrgId": 1}
    )
    return render_template(
        'index.html',
        issues=issues,
        orgs=orgs,
        users=users,
        current_user=current_user,
        db_status=db_status,
        models=AVAILABLE_MODELS,
        selected_model="gpt-5-6-terra"
    )


@app.route('/api/issues', methods=['GET', 'POST'])
def api_issues():
    """GET: List AndOff issues with Status | POST: Create a new AndOff issue with Status."""
    if request.method == 'GET':
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT IssueId, OrgId, IssueTypeId, SubIssueType, PriorityRank, IssueDesc, PartNum, IssuePoc, IssuePocEmail, Status
                        FROM {ISSUES_TABLE}
                        ORDER BY PriorityRank ASC, IssueId DESC
                        LIMIT 100
                    """)
                    rows = cursor.fetchall()
                    return jsonify([
                        {
                            "IssueId": r[0], "OrgId": r[1], "IssueTypeId": r[2],
                            "SubIssueType": r[3], "PriorityRank": r[4], "IssueDesc": r[5],
                            "PartNum": r[6], "IssuePoc": r[7], "IssuePocEmail": r[8],
                            "Status": r[9] if r[9] is not None else "Open"
                        }
                        for r in rows
                    ])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # POST: Insert new issue from 5-Step Wizard
    data = request.get_json(silent=True) or request.form
    issue_id = data.get("IssueId") or f"ISS-{int(datetime.utcnow().timestamp()) % 100000}"
    org_id = str(data.get("OrgId", "1"))
    issue_type = data.get("IssueTypeId", "Mechanical")
    sub_issue_type = data.get("SubIssueType", "")
    priority_rank = int(data.get("PriorityRank", 1))
    issue_desc = data.get("IssueDesc", "")
    part_num = data.get("PartNum", "")
    issue_poc = data.get("IssuePoc", "")
    issue_poc_email = data.get("IssuePocEmail") or get_logged_in_user_email()
    status = data.get("Status", "Open")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {ISSUES_TABLE}
                    (IssueId, OrgId, IssueTypeId, SubIssueType, PriorityRank, IssueDesc, PartNum, IssuePoc, IssuePocEmail, Status, _rescued_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    [issue_id, org_id, issue_type, sub_issue_type, priority_rank, issue_desc, part_num, issue_poc, issue_poc_email, status]
                )
        return jsonify({"status": "created", "IssueId": issue_id, "issue_status": status}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat_with_agent():
    data = request.get_json(silent=True) or {}
    model_id = data.get('model') or request.form.get('model', 'gpt-5-6-terra')
    user_prompt = (data.get('prompt') or request.form.get('prompt', '')).strip()
    if not user_prompt:
        return jsonify({"error": "Please enter a prompt for the agent."}), 400

    if "chat_session_id" not in session:
        session["chat_session_id"] = str(uuid.uuid4())
        session.modified = True
    session_id = session["chat_session_id"]

    if user_prompt.lower() in ("clear", "reset", "/clear", "/reset"):
        save_chat_history(session_id, [])
        return jsonify({
            "model": f"databricks-{model_id}",
            "response": "Conversation history has been cleared. How can I help you with AndOff issues?"
        })

    if "send_tier1_sql_alert" in user_prompt.lower() or "send tier 1 alert" in user_prompt.lower():
        alert_result = send_tier1_sql_alert()
        return jsonify({
            "model": f"databricks-{model_id}",
            "response": alert_result
        })

    chat_history = load_chat_history(session_id)
    reply = ask_databricks_agent(prompt_text=user_prompt, model_id=model_id, chat_history=chat_history)
    chat_history.append({"role": "user", "content": user_prompt})
    chat_history.append({"role": "assistant", "content": str(reply)})
    save_chat_history(session_id, chat_history)

    return jsonify({"model": f"databricks-{model_id}", "response": reply})


# =====================================================================
# --- MCP Server (Mounted at /mcp for AI Agents) ---
# =====================================================================

mcp_server = FastMCP("andoff-mcp-server")


def _run_mcp_sql(query: str) -> str:
    w = WorkspaceClient()
    response = w.statement_execution.execute_statement(
        warehouse_id=get_warehouse_id(),
        statement=query,
        wait_timeout="30s"
    )
    if response.status.state == StatementState.SUCCEEDED:
        if not response.manifest or not response.result or not response.result.data_array:
            return "Query succeeded (no rows returned)."
        columns = [col.name for col in response.manifest.schema.columns]
        rows = response.result.data_array
        header = " | ".join(columns)
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    return f"Query failed: {response.status.error}"


@mcp_server.tool()
def query_catalog_table(query: str) -> str:
    """Run a read-only SQL query against Unity Catalog tables."""
    return _run_mcp_sql(query)


@mcp_server.tool()
def get_andoff_issues(limit: int = 20) -> str:
    """Get AndOff aircraft maintenance issues ordered by PriorityRank, including Status."""
    return _run_mcp_sql(
        f"SELECT IssueId, OrgId, IssueTypeId, SubIssueType, PriorityRank, Status, IssueDesc, PartNum, IssuePoc "
        f"FROM {ISSUES_TABLE} ORDER BY PriorityRank ASC, IssueId DESC LIMIT {int(limit)}"
    )


@mcp_server.tool()
def search_andoff_issues(keyword: str) -> str:
    """Search AndOff issues by keyword in IssueDesc, IssueTypeId, SubIssueType, PartNum, or Status."""
    safe_kw = keyword.replace("'", "''")
    return _run_mcp_sql(
        f"SELECT IssueId, OrgId, IssueTypeId, SubIssueType, PriorityRank, Status, IssueDesc, PartNum, IssuePoc "
        f"FROM {ISSUES_TABLE} "
        f"WHERE IssueDesc ILIKE '%{safe_kw}%' OR IssueTypeId ILIKE '%{safe_kw}%' "
        f"OR SubIssueType ILIKE '%{safe_kw}%' OR PartNum ILIKE '%{safe_kw}%' "
        f"OR Status ILIKE '%{safe_kw}%' "
        f"ORDER BY PriorityRank ASC LIMIT 25"
    )


@mcp_server.tool()
def get_organizations() -> str:
    """List all organizations from t_org_list."""
    return _run_mcp_sql(f"SELECT OrgId, Org, OrgDesc FROM {ORGS_TABLE} ORDER BY OrgId ASC")


@mcp_server.tool()
def get_user_accounts() -> str:
    """List active user accounts and their roles from t_user_accounts."""
    return _run_mcp_sql(
        f"SELECT UserAccountId, DisplayName, Email, Role, OrgId FROM {USERS_TABLE} WHERE IsActive = 1 ORDER BY DisplayName ASC"
    )


@mcp_server.tool()
def send_tier1_sql_alert(job_id: str = "") -> str:
    """Trigger the Databricks Job to evaluate the Tier 1 SQL Alert and send email notifications."""
    target_job_id = job_id.strip() or os.environ.get("DATABRICKS_JOB_ID")
    if not target_job_id:
        return "Error: DATABRICKS_JOB_ID is not set in .env and no job_id was provided."

    url = f"https://{get_server_hostname()}/api/2.1/jobs/run-now"
    payload = json.dumps({"job_id": int(target_job_id)}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_pat_token()}"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return f"Successfully triggered Tier 1 SQL Alert Job (run_id: {data.get('run_id')}). Email will be sent if Tier 1 tickets are open."
    except Exception as e:
        return f"Failed to trigger Job ({target_job_id}): {e}"





# =====================================================================
# --- Unified ASGI Entrypoint (Serves /mcp AND Flask / on Port 8000) ---
# =====================================================================

main_app = FastAPI(title="AndOff Web App & MCP Server")
main_app.mount("/mcp", mcp_server.streamable_http_app())
main_app.mount("/", WSGIMiddleware(app))

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('DATABRICKS_APP_PORT', 8000))
    uvicorn.run(main_app, host='0.0.0.0', port=port)
