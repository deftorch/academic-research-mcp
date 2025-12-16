"""
Academic Research MCP Server - Phase 7: Deep Research Mode
==========================================================

AUTONOMOUS RECURSIVE RESEARCH AGENT

Features:
- Recursive citation traversal (follows rabbit holes)
- Auto-hypothesis testing (finds counter-arguments)
- Multi-perspective analysis (balanced view)
- Critical thinking (peer review mode)
- Autonomous decision making (no hand-holding)
- Safeguards (Circuit Breaker, Rate Limiting)
- Semantic Gap Detection using SentenceTransformers
"""

import logging
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
import asyncio
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from mcp_instance import mcp
from models import Paper
from research_tools import search_all_sources, deduplicate_papers, assess_paper_quality, search_semantic_scholar
from utils import RateLimitExceeded

logger = logging.getLogger(__name__)

# ============================================================================
# MODELS
# ============================================================================

_transformer_model = None

def get_model():
    """Lazy load the sentence transformer model."""
    global _transformer_model
    if _transformer_model is None and _HAS_TRANSFORMERS:
        try:
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
    return _transformer_model

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Simple circuit breaker to stop requests after consecutive failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now().timestamp()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker OPENED due to consecutive failures")

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def is_open(self) -> bool:
        if self.state == "CLOSED":
            return False

        if self.state == "OPEN":
            if (datetime.now().timestamp() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF-OPEN"
                return False  # Allow one request to try
            return True

        return False # HALF-OPEN -> allow

# Global circuit breaker instance
_circuit_breaker = CircuitBreaker()

# ============================================================================
# TOOL 1: DEEP RESEARCH MODE (RECURSIVE)
# ============================================================================

@mcp.tool()
async def deep_research(
    initial_query: str,
    max_depth: int = 3,
    max_papers_per_level: int = 10,
    max_total_requests: int = 50,
    include_counter_arguments: bool = True,
    critical_analysis: bool = True
) -> Dict[str, Any]:
    """
    Autonomous recursive research that follows citation chains.

    Args:
        initial_query: The research topic
        max_depth: How many levels of citation recursion to perform
        max_papers_per_level: Max papers to analyze per level
        max_total_requests: Safety limit for total API requests
        include_counter_arguments: Whether to actively search for opposing views
        critical_analysis: Whether to perform AI peer review on findings
    """
    logger.info(f"Starting deep research: '{initial_query}' (depth: {max_depth})")
    
    if _circuit_breaker.is_open():
        return {
            "error": "Circuit breaker is OPEN. Too many recent API failures. Please try again later.",
            "status": "failed"
        }

    start_time = datetime.now()
    request_count = 0
    
    # Initialize research state
    research_state: Dict[str, Any] = {
        'initial_query': initial_query,
        'max_depth': max_depth,
        'levels': [],
        'papers_analyzed': 0,
        'questions_generated': 0,
        'counter_arguments_found': 0,
        'synthesis': {}
    }
    
    # Check budget
    def check_budget():
        if request_count >= max_total_requests:
            logger.warning(f"Max total requests ({max_total_requests}) reached. Stopping recursion.")
            return False
        if _circuit_breaker.is_open():
            logger.warning("Circuit breaker opened during execution. Stopping.")
            return False
        return True

    # Level 1: Initial search
    logger.info(f"LEVEL 1: Initial search for '{initial_query}'")
    
    try:
        request_count += 1
        level1_papers = await search_all_sources(
            initial_query,
            max_results_per_source=max_papers_per_level
        )
        _circuit_breaker.record_success()
    except Exception as e:
        _circuit_breaker.record_failure()
        logger.error(f"Level 1 search failed: {e}")
        return {"error": str(e), "status": "failed"}
    
    all_papers_level1 = []
    for papers in level1_papers.values():
        all_papers_level1.extend(papers)
    
    # Deduplicate and assess quality
    unique_papers = await deduplicate_papers(all_papers_level1)
    scored_papers = await assess_paper_quality(unique_papers)
    
    # Take top papers
    level1_top = scored_papers[:max_papers_per_level]
    
    research_state['levels'].append({
        'level': 1,
        'query': initial_query,
        'papers_found': len(level1_top),
        # Convert to dict for response serialization
        'top_papers': [p.to_dict() for p in level1_top]
    })
    
    research_state['papers_analyzed'] += len(level1_top)
    
    # Extract research questions from level 1 papers
    follow_up_questions = extract_research_questions(level1_top, initial_query)
    research_state['questions_generated'] += len(follow_up_questions)
    
    logger.info(f"Level 1 complete: {len(level1_top)} papers, {len(follow_up_questions)} follow-up questions")
    
    # Level 2+: Recursive follow-up
    all_follow_up_papers = []
    
    for current_level in range(2, max_depth + 1):
        if not follow_up_questions:
            logger.info(f"No more questions at level {current_level}, stopping recursion")
            break
        
        if not check_budget():
            break

        logger.info(f"LEVEL {current_level}: Following up on {len(follow_up_questions)} questions")
        
        # Limit to top 5 questions to control parallel load
        current_questions = follow_up_questions[:5]
        
        async def fetch_question(q):
            if not check_budget():
                 return []
            try:
                # We increment manually here, but note that race conditions in updating request_count
                # might occur. For simple limiting, this is acceptable.
                # In strict envs, use asyncio.Lock() or atomic counter.
                return await search_semantic_scholar(q, max_results=5)
            except Exception as e:
                logger.warning(f"Failed to search question '{q}': {e}")
                return []

        # Execute searches in parallel
        results = await asyncio.gather(*[fetch_question(q) for q in current_questions])

        # Update metrics (approximated)
        request_count += len(current_questions)
        _circuit_breaker.record_success() # Assume success if we got here

        level_papers = []
        for res in results:
             level_papers.extend(res)
        
        # Deduplicate
        unique = await deduplicate_papers(level_papers)
        
        if unique:
            all_follow_up_papers.extend(unique)
            
            research_state['levels'].append({
                'level': current_level,
                'questions': follow_up_questions[:5],
                'papers_found': len(unique),
                'papers': [p.to_dict() for p in unique]
            })
            
            research_state['papers_analyzed'] += len(unique)
            
            # Generate new questions for next level
            follow_up_questions = extract_research_questions(unique, initial_query)
            research_state['questions_generated'] += len(follow_up_questions)
        else:
            break
    
    # Search for counter-arguments
    counter_papers = [] # Initialize here to be safe
    unique_counter = []

    if include_counter_arguments and check_budget():
        logger.info("Searching for counter-arguments and limitations...")
        
        counter_queries = generate_counter_queries(initial_query)
        
        async def fetch_counter(q):
            if not check_budget():
                return []
            try:
                return await search_semantic_scholar(q, max_results=5)
            except Exception:
                return []

        results = await asyncio.gather(*[fetch_counter(q) for q in counter_queries])
        request_count += len(counter_queries)
        
        for res in results:
            counter_papers.extend(res)

        if counter_papers:
            unique_counter = await deduplicate_papers(counter_papers)
            research_state['counter_arguments'] = {
                'queries': counter_queries,
                'papers_found': len(unique_counter),
                'papers': [p.to_dict() for p in unique_counter]
            }
            research_state['counter_arguments_found'] = len(unique_counter)
    
    # Critical analysis
    if critical_analysis:
        logger.info("Performing critical analysis...")
        
        all_papers = level1_top + all_follow_up_papers
        if include_counter_arguments and counter_papers:
            all_papers.extend(unique_counter)
        
        critique = await perform_critical_analysis(all_papers, initial_query)
        research_state['critical_analysis'] = critique
    
    # Synthesize findings
    logger.info("Synthesizing findings...")
    
    synthesis = synthesize_deep_research(research_state)
    research_state['synthesis'] = synthesis
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()
    research_state['duration_seconds'] = round(duration, 2)
    research_state['total_requests'] = request_count
    
    logger.info(
        f"Deep research complete: {research_state['papers_analyzed']} papers analyzed, "
        f"{len(research_state['levels'])} levels, {duration:.1f}s"
    )
    
    return research_state

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_research_questions(papers: List[Paper], original_query: str) -> List[str]:
    """
    Extract follow-up research questions from papers using semantic gap detection if available.
    """
    model = get_model()
    questions = set()
    
    # Anchor sentences representing gaps/limitations
    gap_anchors = [
        "However, this approach fails to address",
        "A major limitation of this study is",
        "It remains unclear whether",
        "Future work should investigate",
        "This method struggles with",
        "Little is known about"
    ]

    gap_embeddings = None
    if model:
        gap_embeddings = model.encode(gap_anchors)

    for paper in papers[:5]:  # Analyze top 5
        title = paper.title or ""
        abstract = paper.abstract or ""
        text = f"{title}. {abstract}"

        sentences = text.split('. ')
        
        if model and gap_embeddings is not None and len(sentences) > 0:
            # Vector-based Semantic Gap Detection
            sentence_embeddings = model.encode(sentences)

            # Calculate similarity matrix (sentences x anchors)
            similarities = util.cos_sim(sentence_embeddings, gap_embeddings)

            # Find sentences that are semantically similar to gap anchors
            # Threshold 0.3 is arbitrary, tune as needed
            for idx, score in enumerate(np.max(similarities.numpy(), axis=1)):
                if score > 0.4: # Only if somewhat similar
                    sentence = sentences[idx]

                    # Generate question from the gap sentence
                    # Simple heuristic transformation
                    if "limitation" in sentence.lower():
                        questions.add(f"How can we overcome the limitation that {sentence[:50]}...?")
                    elif "future" in sentence.lower():
                        questions.add(f"What are the outcomes of investigating {sentence[:50]}...?")
                    else:
                         questions.add(f"Investigate: {sentence}")

        # Fallback to keyword matching if no model or as supplement
        gap_indicators = [
            'future work', 'limitation', 'further research',
            'remains unclear', 'open question', 'challenge',
            'requires investigation', 'not well understood'
        ]
        
        text_lower = text.lower()
        for indicator in gap_indicators:
            if indicator in text_lower:
                idx = text_lower.find(indicator)
                context = text[max(0, idx-100):min(len(text), idx+100)]
                
                if 'limitation' in context.lower():
                    questions.add(f"What are the limitations of {title[:50]}?")
                elif 'future work' in context.lower():
                    questions.add(f"What future research directions exist for {original_query}?")
                elif 'unclear' in context.lower():
                    questions.add(f"What remains unclear about {original_query}?")
    
    # Default questions if none found
    if not questions:
        questions.add(f"What are the most cited papers about {original_query}?")
        questions.add(f"What are recent developments in {original_query}?")
    
    return list(questions)[:10]

def generate_counter_queries(query: str) -> List[str]:
    """
    Generate queries to find counter-arguments and criticisms.
    """
    counter_queries = [
        f"limitations of {query}",
        f"criticisms of {query}",
        f"challenges in {query}",
        f"problems with {query}",
        f"drawbacks of {query}",
        f"failures of {query}"
    ]
    
    return counter_queries

async def perform_critical_analysis(papers: List[Paper], query: str) -> Dict[str, Any]:
    """
    Perform peer review style critical analysis.
    """
    analysis: Dict[str, Any] = {
        'total_papers': len(papers),
        'methodology_concerns': [],
        'sample_size_issues': [],
        'potential_biases': [],
        'novelty_assessment': {},
        'consensus_level': 'unknown'
    }
    
    # Check sample sizes
    for paper in papers:
        abstract = (paper.abstract or "").lower()
        
        # Look for small sample sizes
        if 'n=' in abstract or 'n =' in abstract:
            # Extract number
            import re
            matches = re.findall(r'n\s*=\s*(\d+)', abstract)
            if matches:
                n = int(matches[0])
                if n < 30:
                    analysis['sample_size_issues'].append({
                        'paper': paper.title,
                        'sample_size': n,
                        'concern': 'Small sample size (n<30) may limit generalizability'
                    })
    
    # Check for conflicts of interest
    for paper in papers:
        authors = paper.authors
        
        # Check for corporate affiliations (crude check)
        corp_keywords = ['google', 'meta', 'openai', 'microsoft', 'amazon', 'facebook']
        for author in authors:
            author_lower = str(author).lower()
            for corp in corp_keywords:
                if corp in author_lower:
                    analysis['potential_biases'].append({
                        'paper': paper.title,
                        'type': 'corporate_affiliation',
                        'detail': f'Author affiliated with {corp.title()}'
                    })
                    break
    
    # Assess consensus
    # If top papers have similar conclusions, there's consensus
    citation_counts = [p.citation_count for p in papers]
    avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0
    
    highly_cited = [p for p in papers if p.citation_count > avg_citations * 1.5]
    
    if len(highly_cited) >= 3:
        analysis['consensus_level'] = 'strong'
    elif len(highly_cited) >= 1:
        analysis['consensus_level'] = 'moderate'
    else:
        analysis['consensus_level'] = 'weak'
    
    # Novelty assessment
    years = [p.year for p in papers if p.year]
    if years:
        valid_years = sorted(years)
        
        if valid_years:
            max_year = valid_years[-1]
            recent_papers = [
                p for p in papers
                if p.year and p.year >= max_year - 2
            ]

            analysis['novelty_assessment'] = {
                'recent_papers': len(recent_papers),
                'oldest_paper': valid_years[0],
                'newest_paper': max_year,
                'field_maturity': 'emerging' if len(papers) < 50 else 'established'
            }
    
    return analysis

def synthesize_deep_research(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize all findings into coherent summary.
    """
    synthesis: Dict[str, Any] = {
        'main_findings': [],
        'evidence_strength': 'unknown',
        'limitations': [],
        'recommendations': []
    }
    
    # Summarize each level
    for level in state['levels']:
        level_num = level['level']
        papers_count = level['papers_found']
        
        if level_num == 1:
            synthesis['main_findings'].append(
                f"Initial search found {papers_count} relevant papers on the topic."
            )
        else:
            synthesis['main_findings'].append(
                f"Level {level_num} follow-up research addressed {len(level.get('questions', []))} "
                f"questions and found {papers_count} additional papers."
            )
    
    # Assess evidence strength
    total_papers = state['papers_analyzed']
    
    if total_papers >= 20:
        synthesis['evidence_strength'] = 'strong'
    elif total_papers >= 10:
        synthesis['evidence_strength'] = 'moderate'
    else:
        synthesis['evidence_strength'] = 'limited'
    
    # Include counter-arguments
    if state.get('counter_arguments_found', 0) > 0:
        synthesis['limitations'].append(
            f"Found {state['counter_arguments_found']} papers discussing "
            f"limitations or counter-arguments."
        )
    
    # Critical analysis insights
    if 'critical_analysis' in state:
        crit = state['critical_analysis']
        
        if crit['sample_size_issues']:
            synthesis['limitations'].append(
                f"{len(crit['sample_size_issues'])} papers have small sample sizes "
                f"that may limit generalizability."
            )
        
        if crit['potential_biases']:
            synthesis['limitations'].append(
                f"{len(crit['potential_biases'])} papers have potential conflicts "
                f"of interest (corporate affiliations)."
            )
        
        consensus = crit.get('consensus_level', 'unknown')
        synthesis['main_findings'].append(
            f"Research consensus: {consensus.upper()}"
        )
    
    # Recommendations
    if synthesis['evidence_strength'] == 'strong':
        synthesis['recommendations'].append(
            "Sufficient evidence exists to draw conclusions. "
            "Consider writing a comprehensive literature review."
        )
    elif synthesis['evidence_strength'] == 'moderate':
        synthesis['recommendations'].append(
            "Moderate evidence. Consider expanding search to more sources or "
            "conducting additional follow-up research."
        )
    else:
        synthesis['recommendations'].append(
            "Limited evidence. This may be an emerging or niche area. "
            "Consider conducting primary research."
        )
    
    return synthesis

# ============================================================================
# TOOL 2: AUTO PEER REVIEW
# ============================================================================

@mcp.tool()
async def peer_review_paper(
    # Can accept either dict or Paper, but likely receives dict via API
    paper: Dict[str, Any],
    review_criteria: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Automatically review a paper using peer review criteria.
    """
    if review_criteria is None:
        review_criteria = [
            'methodology',
            'novelty',
            'clarity',
            'reproducibility',
            'ethical_considerations'
        ]
    
    title = paper.get('title', 'Unknown')
    logger.info(f"Peer reviewing: {title[:50]}...")
    
    review: Dict[str, Any] = {
        'paper_title': title,
        'paper_authors': paper.get('authors', [])[:3],
        'paper_year': paper.get('year'),
        'review_date': datetime.now().isoformat(),
        'criteria_scores': {},
        'overall_score': 0,
        'strengths': [],
        'weaknesses': [],
        'recommendation': 'unknown'
    }
    
    abstract = str(paper.get('abstract', '')).lower()
    title_lower = str(title).lower()
    text = f"{title_lower} {abstract}"
    
    # Methodology check
    if 'methodology' in review_criteria:
        method_score = 5  # Default
        comments = []
        
        if 'experiment' in text or 'study' in text:
            method_score += 2
            comments.append("Includes empirical evaluation")
        
        if 'dataset' in text:
            method_score += 1
            comments.append("Uses dataset(s)")
        
        if 'baseline' in text or 'comparison' in text:
            method_score += 1
            comments.append("Includes baseline comparisons")
        else:
            comments.append("Missing baseline comparisons")
        
        if 'statistical' in text or 'significance' in text:
            method_score += 1
            comments.append("Includes statistical analysis")
        
        review['criteria_scores']['methodology'] = {
            'score': min(method_score, 10),
            'comments': comments
        }
    
    # Novelty check
    if 'novelty' in review_criteria:
        novelty_score = 5
        comments = []
        
        citation_count = int(paper.get('citation_count', 0))
        year = paper.get('year')
        
        if citation_count > 100:
            novelty_score += 2
            comments.append("Highly cited (>100 citations) suggests impact")
        elif citation_count > 10:
            novelty_score += 1
        
        if year:
            try:
                year_int = int(year)
                if year_int >= datetime.now().year - 2:
                    novelty_score += 2
                    comments.append("Recent work (last 2 years)")
            except:
                pass
        
        if 'novel' in text or 'new' in text:
            novelty_score += 1
            comments.append("Claims novelty")
        
        review['criteria_scores']['novelty'] = {
            'score': min(novelty_score, 10),
            'comments': comments
        }
    
    # Clarity check
    if 'clarity' in review_criteria:
        clarity_score = 7  # Default assume clear
        comments = ["Based on title and abstract assessment"]
        
        # Check abstract length (too short or too long)
        if len(abstract) < 100:
            clarity_score -= 2
            comments.append("Very short abstract may lack detail")
        elif len(abstract) > 2000:
            clarity_score -= 1
            comments.append("Very long abstract may lack focus")
        
        review['criteria_scores']['clarity'] = {
            'score': max(clarity_score, 1),
            'comments': comments
        }
    
    # Reproducibility check
    if 'reproducibility' in review_criteria:
        repro_score = 3  # Default low (can't check from abstract alone)
        comments = []
        
        if 'code' in text or 'github' in text or 'available' in text:
            repro_score += 4
            comments.append("Mentions code/resource availability")
        
        if 'open source' in text or 'open-source' in text:
            repro_score += 2
            comments.append("Open source mentioned")
        
        if paper.get('pdf_url'):
            repro_score += 1
            comments.append("PDF publicly available")
        
        review['criteria_scores']['reproducibility'] = {
            'score': min(repro_score, 10),
            'comments': comments
        }
    
    # Calculate overall score
    scores = [c['score'] for c in review['criteria_scores'].values()]
    review['overall_score'] = round(sum(scores) / len(scores), 1) if scores else 0
    
    # Generate recommendation
    if review['overall_score'] >= 8:
        review['recommendation'] = 'Accept'
        review['strengths'].append("High quality across all criteria")
    elif review['overall_score'] >= 6:
        review['recommendation'] = 'Minor Revisions'
        review['strengths'].append("Generally solid work")
        review['weaknesses'].append("Some aspects could be strengthened")
    elif review['overall_score'] >= 4:
        review['recommendation'] = 'Major Revisions'
        review['weaknesses'].append("Significant improvements needed")
    else:
        review['recommendation'] = 'Reject'
        review['weaknesses'].append("Does not meet publication standards")
    
    logger.info(f"Peer review complete: {review['recommendation']} ({review['overall_score']}/10)")
    
    return review
