from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

# This file is kept for backward compatibility if any external scripts rely on it,
# but it now imports from the new core domain location.
from academic_research.domain.models import Paper as CorePaper

class Paper(CorePaper):
    pass
