from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

class Paper(BaseModel):
    """
    Standardized Paper model for all search providers.
    """
    paper_id: Optional[str] = Field(None, description="Unique ID (S2/DOI/ArXiv ID)")
    title: str = Field(..., description="Title of the paper")
    abstract: Optional[str] = Field(None, description="Abstract of the paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    citation_count: int = Field(0, description="Number of citations")
    venue: Optional[str] = Field(None, description="Publication venue")
    pdf_url: Optional[str] = Field(None, description="URL to the PDF")
    source: str = Field(..., description="Source of the data (e.g., 'arXiv', 'Semantic Scholar')")
    url: Optional[str] = Field(None, description="URL to the paper landing page")

    # Provider specific IDs
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    # Additional metadata
    published: Optional[Any] = None # ArXiv published date (struct_time or string)
    categories: List[str] = Field(default_factory=list)
    influential_citation_count: int = 0

    # Quality assessment fields
    quality_score: int = 0
    quality_tier: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility and API response."""
        return self.model_dump(mode='json', exclude_none=True)
