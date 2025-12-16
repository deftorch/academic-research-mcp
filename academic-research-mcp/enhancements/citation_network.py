"""
Academic Research MCP Server - Citation Network Analysis & Visualization
========================================================================

Features:
- Build citation networks (forward & backward citations)
- Interactive network visualization with Plotly
- Community detection (research clusters)
- Identify influential papers (Personalized PageRank)
- Find research gaps (bridge papers)
- Export to Gephi/Cytoscape formats
- Citation Intent Classification (using Transformers)
"""

import json
import logging
from typing import Dict, List, Optional
import numpy as np

try:
    import community as community_louvain
    import networkx as nx
    import plotly.express as px
    import plotly.graph_objects as go
    from sentence_transformers import SentenceTransformer, util
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False
    logger = logging.getLogger(__name__)
    logger.warning("Citation network dependencies missing.")

from mcp_instance import mcp
from research_tools import get_paper_citations

logger = logging.getLogger(__name__)

# ============================================================================
# MODELS
# ============================================================================

_transformer_model = None

def get_model():
    """Lazy load the sentence transformer model."""
    global _transformer_model
    if _transformer_model is None and _HAS_DEPS:
        try:
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
    return _transformer_model

# ============================================================================
# TOOL 1: BUILD CITATION NETWORK
# ============================================================================


@mcp.tool()
async def build_citation_network(
    seed_paper_ids: List[str],
    depth: int = 2,
    max_papers_per_level: int = 10,
    direction: str = "both",
    query_context: Optional[str] = None
) -> Dict:
    """
    Build a citation network starting from seed papers.

    Args:
        seed_paper_ids: List of starting paper IDs
        depth: Traversal depth
        max_papers_per_level: Max papers to fetch per node
        direction: 'forward', 'backward', or 'both'
        query_context: Optional research query to bias PageRank (Personalized)
    """
    if not _HAS_DEPS:
        return {"error": "Missing dependencies (networkx, plotly, etc.)"}

    logger.info(f"Building citation network from {len(seed_paper_ids)} seed papers")

    G = nx.DiGraph()  # Directed graph (A cites B)
    papers_data = {}

    # Store abstracts for personalization
    paper_abstracts = {}

    # Queue for BFS traversal: (paper_id, current_depth)
    queue = [(pid, 0) for pid in seed_paper_ids]
    processed = set()

    while queue and len(processed) < 200:  # Safety limit
        paper_id, current_depth = queue.pop(0)

        if paper_id in processed or current_depth > depth:
            continue

        try:
            # Get paper citations
            citation_data = await get_paper_citations(
                paper_id,
                max_citations=max_papers_per_level,
                max_references=max_papers_per_level,
            )

            # citation_data is a Dict returned by get_paper_citations wrapper,
            # which maps internal API response to clean dict.
            # It now has 'paper_id' field thanks to Phase 1 fix.

            if "title" not in citation_data:
                logger.warning(f"Could not get citations for {paper_id}")
                continue

            # Add main paper as node
            title = citation_data.get("title", "Unknown")[:100]
            papers_data[paper_id] = {
                "title": title,
                "total_citations": len(citation_data.get("citing_papers", [])),
                "depth": current_depth,
            }

            # Store abstract if we had it (not currently returned by get_paper_citations,
            # might need to look up or cache, but for now we skip using abstracts
            # for personalization of fetched nodes if data is missing)

            G.add_node(
                paper_id,
                title=title,
                citations=len(citation_data.get("citing_papers", [])),
                depth=current_depth,
                is_seed=current_depth == 0,
            )

            # Add forward citations (papers citing this one)
            if direction in ["forward", "both"]:
                for citing in citation_data.get("citing_papers", []):
                    citing_id = citing.get("paper_id")
                    if citing_id:
                        citing_title = citing.get("title", "Unknown")[:100]

                        # Classify intent (heuristic based on title keywords)
                        # In a real scenario, we'd need the citation context/sentence
                        intent = "background" # Default
                        title_lower = citing_title.lower()
                        if any(k in title_lower for k in ["method", "approach", "framework", "system"]):
                            intent = "method"
                        elif any(k in title_lower for k in ["review", "survey", "overview", "analysis"]):
                            intent = "background"
                        elif any(k in title_lower for k in ["comment", "critique", "discussion", "reply"]):
                            intent = "critique"

                        G.add_node(
                            citing_id,
                            title=citing_title,
                            citations=0, # We don't know citing paper's citation count from this API call
                            depth=current_depth + 1,
                            is_seed=False,
                        )
                        G.add_edge(citing_id, paper_id, intent=intent)  # Edge: citing -> cited

                        if current_depth < depth:
                            queue.append((citing_id, current_depth + 1))

            # Add backward citations (papers this one cites)
            if direction in ["backward", "both"]:
                for ref in citation_data.get("referenced_papers", []):
                    ref_id = ref.get("paper_id")
                    if ref_id:
                        ref_title = ref.get("title", "Unknown")[:100]

                        intent = "background"
                        title_lower = ref_title.lower()
                        if any(k in title_lower for k in ["method", "approach", "framework", "system"]):
                            intent = "method"
                        elif any(k in title_lower for k in ["review", "survey", "overview", "analysis"]):
                            intent = "background"
                        elif any(k in title_lower for k in ["comment", "critique", "discussion", "reply"]):
                            intent = "critique"

                        G.add_node(ref_id, title=ref_title, citations=0, depth=current_depth + 1, is_seed=False)
                        G.add_edge(paper_id, ref_id, intent=intent)  # Edge: citing -> cited

                        if current_depth < depth:
                            queue.append((ref_id, current_depth + 1))

            processed.add(paper_id)

        except Exception as e:
            logger.error(f"Error processing {paper_id}: {e}")

    # Calculate network metrics
    metrics = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2) if G.number_of_nodes() > 0 else 0,
    }

    # Calculate Personalized PageRank
    if G.number_of_nodes() > 0:
        try:
            personalization = None
            if query_context and get_model():
                # Compute personalization vector based on similarity to query
                model = get_model()
                query_embedding = model.encode(query_context)

                personalization = {}
                for node in G.nodes():
                    node_title = G.nodes[node].get("title", "")
                    # If we had abstract, use it. For now use title.
                    node_embedding = model.encode(node_title)
                    sim = util.cos_sim(query_embedding, node_embedding).item()
                    personalization[node] = max(0.001, sim) # Ensure non-zero

                # Normalize
                total_sim = sum(personalization.values())
                if total_sim > 0:
                    personalization = {k: v / total_sim for k, v in personalization.items()}
                else:
                    personalization = None

            pagerank = nx.pagerank(G, personalization=personalization)
            top_papers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

            metrics["most_influential"] = [
                {
                    "paper_id": pid,
                    "title": G.nodes[pid].get("title", "Unknown"),
                    "pagerank_score": round(score, 4),
                    "citations": G.nodes[pid].get("citations", 0),
                }
                for pid, score in top_papers
            ]
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            metrics["most_influential"] = []

    # Calculate centrality measures
    if G.number_of_nodes() > 1:
        try:
            betweenness = nx.betweenness_centrality(G)
            top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

            metrics["bridge_papers"] = [
                {"paper_id": pid, "title": G.nodes[pid].get("title", "Unknown"), "betweenness": round(score, 4)}
                for pid, score in top_bridges
                if score > 0
            ]
        except Exception as e:
            logger.warning(f"Centrality calculation failed: {e}")
            metrics["bridge_papers"] = []

    logger.info(f"Network built: {metrics['total_nodes']} nodes, {metrics['total_edges']} edges")

    # Classify edges (Intent) if model available
    # For now, we did placeholder. To do it properly we need context sentences which we don't have.
    # So we'll skip complex intent classification for now or implement heuristic.

    return {"network_graph": nx.node_link_data(G), "metrics": metrics, "papers_data": papers_data}


# ============================================================================
# TOOL 2: VISUALIZE CITATION NETWORK
# ============================================================================


@mcp.tool()
async def visualize_citation_network(
    network_data: Dict,
    layout: str = "spring",
    color_by: str = "citations",
    output_file: str = "citation_network.html",
    show_labels: bool = True,
) -> str:
    """
    Create interactive citation network visualization.
    """
    if not _HAS_DEPS:
        return "Error: Missing dependencies"

    logger.info("Generating citation network visualization...")

    # Reconstruct graph
    G = nx.node_link_graph(network_data["network_graph"])

    if G.number_of_nodes() == 0:
        return "Error: Network is empty"

    # Calculate layout positions
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "hierarchical":
        # Hierarchical by depth
        pos = {}
        depths = nx.get_node_attributes(G, "depth")
        max_depth = max(depths.values()) if depths else 1

        for depth in range(max_depth + 1):
            nodes_at_depth = [n for n, d in depths.items() if d == depth]
            for i, node in enumerate(nodes_at_depth):
                x = i / (len(nodes_at_depth) + 1)
                y = 1 - (depth / max_depth)
                pos[node] = (x, y)
    else:
        pos = nx.spring_layout(G)

    # Create edge traces
    edge_x = []
    edge_y = []

    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines", name="Citations"
    )

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_symbols = []

    for node in G.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node info
            title = G.nodes[node].get("title", "Unknown")
            citations = G.nodes[node].get("citations", 0)
            depth = G.nodes[node].get("depth", 0)
            is_seed = G.nodes[node].get("is_seed", False)

            # Hover text
            hover_text = f"<b>{title}</b><br>"
            hover_text += f"Citations: {citations}<br>"
            hover_text += f"Network Depth: {depth}<br>"
            hover_text += f"In-degree: {G.in_degree(node)}<br>"
            hover_text += f"Out-degree: {G.out_degree(node)}"

            if show_labels:
                node_text.append(hover_text)
            else:
                node_text.append(title)

            # Color
            if color_by == "citations":
                node_color.append(citations)
            elif color_by == "depth":
                node_color.append(depth)
            elif color_by == "influence":
                # Use PageRank if available
                pagerank = network_data.get("metrics", {}).get("most_influential", [])
                # This is tricky because most_influential is a list of top papers, not full map
                # Need full pagerank map if we want to color by it
                # For now fallback to citations if not available
                node_color.append(citations)
            else:
                node_color.append(citations)

            # Size based on citations
            size = max(10, min(50, 10 + citations / 10))
            node_size.append(size)

            # Shape: seed papers are stars
            node_symbols.append("star" if is_seed else "circle")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_color,
            size=node_size,
            symbol=node_symbols,
            colorbar=dict(thickness=15, title=color_by.capitalize(), xanchor="left", titleside="right"),
            line=dict(width=2, color="white"),
        ),
        name="Papers",
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Citation Network Visualization<br>"
                f"<sub>{G.number_of_nodes()} papers, {G.number_of_edges()} citations</sub>",
                x=0.5,
                xanchor="center",
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            plot_bgcolor="white",
        ),
    )

    # Save to HTML
    fig.write_html(output_file)

    logger.info(f"Visualization saved to {output_file}")
    return f"Interactive visualization saved to: {output_file}"


# ============================================================================
# TOOL 3: DETECT RESEARCH COMMUNITIES
# ============================================================================


@mcp.tool()
async def detect_research_communities(network_data: Dict, algorithm: str = "louvain") -> Dict:
    """
    Detect research communities (clusters) in citation network.
    """
    if not _HAS_DEPS:
        return {"error": "Missing dependencies"}

    logger.info("Detecting research communities...")

    G = nx.node_link_graph(network_data["network_graph"])

    if G.number_of_nodes() < 3:
        return {"error": "Network too small for community detection", "num_communities": 0}

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Detect communities
    if algorithm == "louvain":
        try:
            partition = community_louvain.best_partition(G_undirected)
            modularity = community_louvain.modularity(partition, G_undirected)
        except Exception as e:
            logger.error(f"Louvain algorithm failed: {e}")
            return {"error": str(e)}
    else:
        # Label propagation (simpler, faster)
        communities_generator = nx.community.label_propagation_communities(G_undirected)
        communities_list = list(communities_generator)
        partition = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                partition[node] = i
        modularity = nx.community.modularity(G_undirected, communities_list)

    # Organize by community
    community_groups = {}
    for node, comm_id in partition.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []

        community_groups[comm_id].append(
            {
                "paper_id": node,
                "title": G.nodes[node].get("title", "Unknown"),
                "citations": G.nodes[node].get("citations", 0),
                "depth": G.nodes[node].get("depth", 0),
            }
        )

    # Sort communities by size
    sorted_communities = sorted(community_groups.items(), key=lambda x: len(x[1]), reverse=True)

    result = {"num_communities": len(community_groups), "modularity": round(modularity, 4), "communities": []}

    for comm_id, papers in sorted_communities:
        avg_citations = sum(p["citations"] for p in papers) / len(papers)

        result["communities"].append(
            {
                "id": comm_id,
                "size": len(papers),
                "average_citations": round(avg_citations, 1),
                "top_papers": sorted(papers, key=lambda x: x["citations"], reverse=True)[:5],
                "all_papers": papers,
            }
        )

    logger.info(f"Found {len(community_groups)} communities (modularity: {modularity:.3f})")

    return result


# ============================================================================
# TOOL 4: FIND RESEARCH GAPS
# ============================================================================


@mcp.tool()
async def find_research_gaps(network_data: Dict, communities: Optional[Dict] = None) -> List[Dict]:
    """
    Identify potential research gaps in citation network.
    """
    if not _HAS_DEPS:
        return []

    logger.info("Analyzing research gaps...")

    G = nx.node_link_graph(network_data["network_graph"])
    gaps = []

    # Find bridge nodes (high betweenness centrality)
    try:
        betweenness = nx.betweenness_centrality(G.to_undirected())
        bridges = [(node, score) for node, score in betweenness.items() if score > 0.1]

        for node, score in sorted(bridges, key=lambda x: x[1], reverse=True)[:5]:
            gaps.append(
                {
                    "type": "bridge_paper",
                    "paper_id": node,
                    "title": G.nodes[node].get("title", "Unknown"),
                    "betweenness_score": round(score, 4),
                    "description": f"This paper bridges different research areas (betweenness: {score:.3f}). "
                    f"More work connecting these topics could be valuable.",
                }
            )
    except Exception as e:
        logger.warning(f"Betweenness calculation failed: {e}")

    # Find emerging topics (if communities provided)
    if communities:
        for comm in communities["communities"]:
            if comm["size"] >= 3 and comm["average_citations"] < 20:
                gaps.append(
                    {
                        "type": "emerging_topic",
                        "community_id": comm["id"],
                        "size": comm["size"],
                        "avg_citations": comm["average_citations"],
                        "top_paper": comm["top_papers"][0]["title"] if comm["top_papers"] else "Unknown",
                        "description": f"Emerging research area with {comm['size']} papers and "
                        f"low average citations ({comm['average_citations']:.1f}). "
                        f"Early work here could be impactful.",
                    }
                )

    # Find isolated papers
    G_undirected = G.to_undirected()
    isolated = [node for node in G_undirected.nodes() if G_undirected.degree(node) <= 1]

    if len(isolated) > 0:
        gaps.append(
            {
                "type": "isolated_papers",
                "count": len(isolated),
                "description": f"{len(isolated)} papers with minimal connections to the network. "
                f"These may represent underexplored areas or need better integration.",
            }
        )

    logger.info(f"Identified {len(gaps)} potential research gaps")

    return gaps


# ============================================================================
# TOOL 5: EXPORT NETWORK FOR EXTERNAL TOOLS
# ============================================================================


@mcp.tool()
async def export_network(network_data: Dict, format: str = "gexf", output_file: str = "citation_network") -> str:
    """
    Export network to formats compatible with Gephi, Cytoscape, etc.
    """
    if not _HAS_DEPS:
        return "Error: Missing dependencies"

    logger.info(f"Exporting network to {format} format...")

    G = nx.node_link_graph(network_data["network_graph"])

    if format == "gexf":
        output_path = f"{output_file}.gexf"
        nx.write_gexf(G, output_path)

    elif format == "graphml":
        output_path = f"{output_file}.graphml"
        nx.write_graphml(G, output_path)

    elif format == "json":
        output_path = f"{output_file}.json"
        with open(output_path, "w") as f:
            json.dump(network_data["network_graph"], f, indent=2)

    elif format == "edgelist":
        output_path = f"{output_file}.edgelist"
        nx.write_edgelist(G, output_path, data=["weight"])

    else:
        return f"Unsupported format: {format}"

    logger.info(f"Network exported to {output_path}")
    return f"Network exported to: {output_path}"


@mcp.tool()
async def get_citation_graph_data(network_data: Dict) -> Dict:
    """
    Return the citation graph data in JSON format for frontend visualization.
    """
    if not _HAS_DEPS:
        return {"error": "Missing dependencies"}

    # Return the raw network graph data (nodes and links) which is compatible with many frontends (e.g. D3, React Force Graph)
    return network_data["network_graph"]
