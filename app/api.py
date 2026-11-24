from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from swytch import scoring, matching
from swytch.features import PERSON

app = FastAPI(title="Swytch fit API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # tighten in prod

class Values(BaseModel):
    values: dict[str, int]          # keys = PERSON names, each 1..4
    @field_validator("values")
    @classmethod
    def check(cls, v):
        if set(v) != set(PERSON) or not all(1 <= x <= 4 for x in v.values()):
            raise ValueError("values must contain all 9 importance items, each 1-4")
        return v

class EvaluateReq(Values):
    title: str
    description: str = ""
    profile_id: str | None = None   # set when the user corrects the match

class RecommendReq(Values):
    k: int = 10

@app.post("/evaluate")
def evaluate(req: EvaluateReq):
    cands = matching.top_matches(f"{req.title} {req.description}")
    pid = req.profile_id or cands[0][0]
    bd = scoring.breakdown(req.values, pid)
    return {"matched": scoring.occ.loc[pid, "title"], "candidates": cands,
            "low_confidence": cands[0][2] < 0.2,
            "label": scoring.label(req.values, pid), "breakdown": bd, "why": scoring.explain(bd)}

@app.post("/recommend")
def recommend(req: RecommendReq):
    return scoring.recommend(req.values, req.k)