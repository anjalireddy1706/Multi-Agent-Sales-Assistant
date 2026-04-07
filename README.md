# 🤖 Multi-Agent Sales Assistant

An AI-powered sales chatbot that doesn't just answer questions — it reads the emotional tone of the conversation and adapts its strategy in real time. Built with a multi-agent architecture, Agentic RAG, and a dedicated security layer.

---

## 💡 The Idea

Most sales bots repeat the same scripted answers regardless of how the customer feels. This project explores whether an AI can adjust its negotiation strategy based on the emotional trajectory of a conversation — just like a skilled human sales rep would.

**Example:** A customer frustrated about pricing doesn't need to hear the MSRP repeated. They need empathy, context, and a value-focused response. This system detects that frustration and responds accordingly.

---

## 🏗️ Architecture

The system uses an **intent-based router** to dynamically orchestrate 5 specialised agents rather than a fixed linear chain.

### Agents

| Agent | Role |
|-------|------|
| 🧭 **Intent Detector** | Classifies user intent (price objection, product inquiry, buying signal, small talk) and labels each message before routing |
| 🔍 **Researcher** | Performs semantic search over the product knowledge base via Agentic RAG to retrieve verified facts |
| 🧠 **Profiler** | Analyses conversation sentiment to detect if the buyer is engaging or pulling away. Activates after 4+ messages to ensure enough signal exists before profiling |
| 🎯 **Closer** | Synthesises retrieved facts and the buyer profile to craft a persuasive, tone-aware response |
| 🛡️ **Auditor** | N-shot self-reflection loop that cross-checks every claim against the source PDF and business rules to eliminate hallucinations |

### Security Layer

A dedicated security wrapper runs **before** every agent call — built entirely in pure Python, no LLM involved:

- Regex-based prompt injection detection (20+ patterns)
- Data extraction attack detection
- Input length and topic scope validation
- Human-in-the-loop flagging for high-risk actions
- Output scanning to block sensitive data leakage
- Full audit logging of every interaction

---

## 🛠️ Tech Stack

- **LLM** — GPT-4o via Azure OpenAI
- **RAG** — ChromaDB + HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Framework** — LangChain + Streamlit
- **Security** — Custom regex-based injection defence (no LLM involved)
- **Language** — Python 3.11+

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/anjalireddy1706/Multi-Agent-Sales-Assistant.git
cd Multi-Agent-Sales-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_azure_api_key
AZURE_API_VERSION=2024-12-01-preview
CHAT_DEPLOYMENT_NAME=gpt-4o
```

### 5. Run the app

```bash
streamlit run app.py
```

### 6. Upload a product PDF

Use the sidebar to upload and index a product specification PDF. The assistant will use this as its knowledge base.

---

## 💬 Demo

### Greeting — intent detected as `GREETING_SMALL_TALK`
The bot responds naturally and opens the conversation without pushing a sale immediately.

![Greeting](screenshots/greeting.png)

---

### Pricing query — intent detected as `PRICING_INQUIRY`
The Researcher agent retrieves the exact pricing from the product PDF. The Closer presents it clearly with a value-led framing rather than just listing numbers.

![Pricing](screenshots/pricing.png)

---

### Multi-item purchase — sentiment detected as `curious`
The customer asks about buying three units. The bot calculates the total, stays helpful, and the Profiler flags the sentiment as curious — adapting the tone accordingly.

![Purchase](screenshots/purchase.png)

---

### Buying signal — sentiment detected as `excited`
The customer signals purchase intent. The bot detects excitement, shifts into closing mode, reinforces value with product stats, and drives toward a decision.

![Buying Signal](screenshots/buying_signal.png)

---

### Prompt injection blocked 🛡️
A user attempts a data extraction attack. The security layer catches it instantly using regex — before any LLM is involved — and blocks it with a clear message.

![Security](screenshots/security.png)

---

## 📁 Project Structure

```
├── app.py                    # Main Streamlit UI and agent pipeline
├── agent_intentdetection.py  # Intent classification
├── agent_profiler.py         # Buyer sentiment profiling
├── agent_researcher.py       # RAG-based product knowledge retrieval
├── agent_closer.py           # Response generation
├── agent_auditor.py          # Hallucination detection and fact checking
├── agent_securitycheck.py    # Prompt injection defence and audit logging
├── .gitignore
└── README.md
```
