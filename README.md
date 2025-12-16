# 🎓 Academic Research MCP Server

A production-ready **Model Context Protocol (MCP)** server designed for autonomous academic research. This tool transforms Claude into a capable research assistant that can search multiple databases, analyze papers, generate citation networks, and perform recursive deep research.

## ⭐ Key Features

### 1. Core System (Phase 1)
* **Multi-Source Search**: Concurrent search across **arXiv**, **Semantic Scholar**, **CrossRef**, and **PubMed**.
* **Intelligent Processing**:
    * **Deduplication**: Merges duplicate papers using fuzzy string matching.
    * **Quality Assessment**: Auto-scores papers (0-12 scale) based on citations, venue quality, and recency.
* **Automation Pipeline**: Single-command `research_pipeline` to search, filter, analyze, and summarize findings.
* **PDF Tools**: Metadata extraction and open access discovery via Unpaywall.

### 2. Advanced Enhancements (Modular)
* **🧠 RAG & Semantic Search**: Index papers into a vector database (ChromaDB) for semantic querying and evidence-based Q&A.
* **🕸️ Citation Network Analysis**: Build interactive graphs to identify influential papers, research gaps, and communities.
* **📈 Trend Analysis**: Forecast research directions and identify "hot" vs "cooling" topics using historical data.
* **🕵️ Deep Research Agent**: Autonomous recursive research that follows citation trails ("rabbit holes") and performs critical peer-review analysis.
* **🔗 Knowledge Management**: Two-way sync with **Zotero** and export to **Notion**.

---

## 📂 Project Structure

```text
academic-research-mcp/
├── academic_research_server.py  # Main Entry Point & Integration
├── utils.py                     # Configuration & Utility functions
├── requirements.txt             # Project dependencies
├── enhancements/                # Modular feature sets
│   ├── rag_enhancement.py       # Vector DB & Semantic Search
│   ├── citation_network.py      # NetworkX & Graph Analysis
│   ├── trend_analysis.py        # Trend Forecasting
│   ├── deep_research.py         # Recursive Research Agent
│   ├── knowledge_management.py  # Zotero/Notion Integration
│   ├── advanced_features.py     # ML Recommendations
│   └── monetization.py          # (Optional) Tier & Rate Limit Logic
└── tests/
    └── test_academic_mcp.py     # Benchmark & Test suite
````

## 📦 Installation

### Prerequisites

  * Python 3.10 or higher
  * pip (Python package manager)
  * (Optional) NVIDIA GPU with CUDA (strongly recommended for marker-pdf and embeddings)

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd academic-research-mcp
```

### 2\. Set Up Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to avoid dependency conflicts.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

> **⚡ GPU Acceleration Note:** If you plan to use the Deep Research or RAG modules with GPU acceleration, install PyTorch with CUDA support *before* installing requirements. Visit pytorch.org for the correct command.

-----

## ⚙️ Configuration

### 1\. Basic Setup

The core configuration lives in the utilities module to allow import across all tools.
Open `utils.py` and edit the following constants:

  * **`CONTACT_EMAIL`**: Required for polite API usage (CrossRef, Unpaywall).
  * **`CACHE_TTL`**: Cache duration in seconds (default: 3600).

### 2\. Module Auto-Detection

The server uses a modular architecture. It automatically detects which libraries are installed:

  * If `chromadb` is missing, RAG features are disabled.
  * If `networkx` is missing, Citation Network features are disabled.

To enable a feature, simply ensure its dependencies are installed via pip.

### 3\. API Keys (Security Best Practice)

Basic search is free. For enhanced features (Zotero, Notion, etc.), it is recommended to set environment variables instead of hardcoding credentials in the source code.

**Linux/macOS:**

```bash
export ZOTERO_API_KEY="your-key"
export NOTION_API_KEY="your-key"
python academic_research_server.py
```

**Windows (PowerShell):**

```powershell
$env:ZOTERO_API_KEY="your-key"
python academic_research_server.py
```

### 4\. Claude Desktop Integration

Add the server to your Claude Desktop configuration file.

  * **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
  * **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

<!-- end list -->

```json
{
  "mcpServers": {
    "academic-research": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": [
        "/absolute/path/to/academic_research_server.py"
      ]
    }
  }
}
```

> **⚠️ Windows User Important Note:** You MUST use **double backslashes** (`\\`) in your JSON paths to avoid syntax errors.
>
> **Example:**
> `"command": "C:\\Users\\Name\\academic-mcp\\venv\\Scripts\\python.exe"`

-----

## 🚀 Usage

### Running the Server

Ensure your virtual environment is activated, then run:

```bash
python academic_research_server.py
```

### Example Prompts (in Claude)

Once connected, you can ask Claude to perform complex research tasks:

  * **Discovery**: "Search for papers on 'Chain of Thought reasoning' published in 2024. Filter for high quality."
  * **Deep Research**: "Perform deep research on 'Limitations of RAG systems'. Follow citation trails to find counter-arguments."
  * **Analysis**: "Build a citation network for the 'Attention Is All You Need' paper and identify the most influential derivative works."
  * **Management**: "Find the top 5 papers on 'Graph Neural Networks' and sync them to my Zotero library."

-----

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Claude "Connection Failed"** | Check `claude_desktop_config.json`. Absolute paths are required. Ensure you point to the python executable *inside* your venv. Windows users check for double backslashes (`\\`). |
| **"Module not found"** | Ensure you activated the virtual environment (`source venv/bin/activate`) before running the server. |
| **GPU not used** | Run `python -c "import torch; print(torch.cuda.is_available())"`. If False, reinstall PyTorch with CUDA support. |
| **Rate Limit Errors** | The server handles backoff automatically. If persistent, consider adding API keys for Semantic Scholar. |

-----

## 🧪 Testing

Run the integrated benchmark suite to verify API connectivity and pipeline performance:

```bash
python tests/test_academic_mcp.py
```

This will generate a `benchmark_results.json` and a markdown report validating your installation.

## 📜 License

MIT License. See LICENSE for details.
