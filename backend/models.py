from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class DocumentChunk:
    page_content: str
    metadata: Dict = field(default_factory=dict)
    # metadata should contain: source (filename), page_number, chunk_id

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
