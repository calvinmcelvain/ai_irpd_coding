from pydantic import BaseModel


# STAGE 0
class Stage_0_Structure(BaseModel):
    window_number: int
    summary: str


# STAGE 1
class Examples(BaseModel):
    window_number: int
    reasoning: str

class Category(BaseModel):
    category_name: str
    definition: str
    examples: list[Examples]

class Stage_1_Structure(BaseModel):
    categories: list[Category]


# STAGE 1r
class Refinement(BaseModel):
    category_name: str
    keep_decision: bool
    reasoning: str

class Stage_1r_Structure(BaseModel):
    final_categories: list[Refinement]


# STAGE 2
class Stage_2_Structure(BaseModel):
    window_number: str
    assigned_categories: list[str]
    reasoning: str


# STAGE 3
class Ranking(BaseModel):
    category_name: str
    rank: int

class Stage_3_Structure(BaseModel):
    window_number: str
    category_ranking: list[Ranking]
    reasoning: str