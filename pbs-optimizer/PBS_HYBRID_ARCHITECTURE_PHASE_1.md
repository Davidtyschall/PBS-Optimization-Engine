# PBS Optimization Engine - Hybrid Architecture
# Phase 1

## Overview

Production-grade parsing system combining regex (fast, free) with LLM parsing (accurate, handles noise) for optimal cost-efficiency and data quality.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                 │
│                   (React / Next.js)                             │
│         - Upload PDF                                            │
│         - View parsed sequences                                 │
│         - Set preferences                                       │
│         - See recommendations                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY                                │
│                  (FastAPI / Flask)                              │
│         - Auth (JWT)                                            │
│         - Rate limiting                                         │
│         - Request validation                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────────┐  ┌─────────────┐
│   PARSER    │  │   CACHE         │  │  OPTIMIZER  │
│   SERVICE   │  │   (Redis)       │  │  SERVICE    │
│             │  │                 │  │             │
│ - OCR       │  │ - Parsed pages  │  │ - Scoring   │
│ - Regex     │  │ - LLM responses │  │ - Ranking   │
│ - LLM calls │  │ - User prefs    │  │ - Recs      │
└──────┬──────┘  └─────────────────┘  └─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM PROVIDER                                │
│              (Anthropic API / OpenAI API)                       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATABASE                                   │
│                    (PostgreSQL)                                 │
│         - Users                                                 │
│         - Parsed bid packets                                    │
│         - Preferences                                           │
│         - Recommendations                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cost Analysis (1,000 users, 50 pages/user/month)

| Component | Provider | Monthly Cost |
|-----------|----------|--------------|
| OCR | Tesseract (free) | $0 |
| LLM Parsing | Claude Haiku / GPT-4o mini | $50-100 |
| Hosting | VPS (Railway/Render) | $20-50 |
| Database | PostgreSQL | $0-20 |
| Cache | Redis | $0-15 |
| **Total** | | **$70-185** |

**Revenue at $50/user:** $50,000/month
**Margin:** 99%+

---

## Parsing Pipeline

### Tier 1: Regex (Free, Fast)
Used for simple, reliable patterns:
- SEQ headers
- RPT times
- RLS times
- TTL totals (with normalization)

### Tier 2: LLM (Paid, Accurate)
Used for complex patterns or when regex fails:
- Flight legs
- Duty periods
- Layover info
- Calendar grid
- Edge cases

### Flow

```python
def parse_page(text: str) -> dict:
    result = {}
    
    # Tier 1: Regex (free)
    result['seq_headers'] = regex_parse_seq(text)
    result['rpt_times'] = regex_parse_rpt(text)
    result['rls_times'] = regex_parse_rls(text)
    result['ttl'] = regex_parse_ttl(text)
    
    # Tier 2: LLM (complex patterns)
    result['flight_legs'] = llm_parse(text, 'flight_legs')
    result['duty_periods'] = llm_parse(text, 'duty_periods')
    result['layovers'] = llm_parse(text, 'layovers')
    
    # Validate and assemble
    return assemble_and_validate(result)
```

---

## Caching Strategy

### Why Cache?
- Same bid packet structures repeat monthly
- Same sequences across users at same base
- After month 1, 80%+ cache hit rate

### Implementation

```python
import hashlib
import json
from functools import lru_cache

# In-memory LRU cache
@lru_cache(maxsize=10000)
def get_cached_parse(text_hash: str):
    return None  # Cache miss handled by caller

# Persistent cache (Redis/DB)
def parse_with_cache(text: str) -> dict:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check memory cache
    cached = get_cached_parse(text_hash)
    if cached:
        return cached
    
    # Check persistent cache
    cached = redis.get(f"parse:{text_hash}")
    if cached:
        return json.loads(cached)
    
    # Parse (regex + LLM)
    result = parse_page(text)
    
    # Store in both caches
    redis.setex(f"parse:{text_hash}", 86400 * 30, json.dumps(result))
    get_cached_parse.cache_clear()  # Update LRU
    
    return result
```

---

## Database Schema

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    airline VARCHAR(50),
    base VARCHAR(10),
    equipment VARCHAR(10),
    seniority_number INT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Parsed bid packets
CREATE TABLE bid_packets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    file_hash VARCHAR(64) UNIQUE,
    filename VARCHAR(255),
    raw_text TEXT,
    parsed_json JSONB,
    page_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Individual sequences (denormalized for fast queries)
CREATE TABLE sequences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bid_packet_id UUID REFERENCES bid_packets(id),
    seq_num INT,
    ops_count INT,
    positions TEXT[],
    total_block VARCHAR(10),
    total_tpay VARCHAR(10),
    tafb VARCHAR(10),
    sequence_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User preferences
CREATE TABLE preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) UNIQUE,
    prefer_weekends_off BOOLEAN DEFAULT false,
    preferred_destinations TEXT[],
    avoid_destinations TEXT[],
    min_layover_hours FLOAT,
    max_tafb_hours FLOAT,
    maximize_pay BOOLEAN DEFAULT false,
    custom_rules JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- LLM response cache
CREATE TABLE llm_cache (
    text_hash VARCHAR(64) PRIMARY KEY,
    pattern_type VARCHAR(50),
    response JSONB,
    tokens_used INT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_bid_packets_user ON bid_packets(user_id);
CREATE INDEX idx_bid_packets_hash ON bid_packets(file_hash);
CREATE INDEX idx_sequences_bid_packet ON sequences(bid_packet_id);
CREATE INDEX idx_sequences_seq_num ON sequences(seq_num);
CREATE INDEX idx_llm_cache_expires ON llm_cache(expires_at);
```

---

## API Endpoints

```python
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import hashlib

app = FastAPI(title="PBS Optimizer API")

# ============================================================
# PARSING ENDPOINTS
# ============================================================

@app.post("/api/v1/parse")
async def parse_bid_packet(
    file: UploadFile,
    user: User = Depends(get_current_user)
):
    """
    Upload and parse a bid packet PDF.
    Returns parsed sequences in JSON format.
    """
    # Rate limiting
    check_rate_limit(user.id, limit=10, window=86400)
    
    # Read and hash file
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    
    # Check cache
    cached = await db.get_bid_packet_by_hash(file_hash)
    if cached:
        return {"status": "cached", "data": cached.parsed_json}
    
    # Extract text via OCR
    text = extract_ocr(content)
    
    # Parse (hybrid regex + LLM)
    parsed = await parse_pipeline(text)
    
    # Validate
    validated = validate_parsed_data(parsed)
    
    # Store
    bid_packet_id = await db.save_bid_packet(
        user_id=user.id,
        file_hash=file_hash,
        filename=file.filename,
        raw_text=text,
        parsed_json=validated
    )
    
    return {
        "status": "parsed",
        "bid_packet_id": bid_packet_id,
        "data": validated
    }


@app.get("/api/v1/bid-packets/{bid_packet_id}")
async def get_bid_packet(
    bid_packet_id: str,
    user: User = Depends(get_current_user)
):
    """Get a previously parsed bid packet."""
    bid_packet = await db.get_bid_packet(bid_packet_id)
    
    if not bid_packet or bid_packet.user_id != user.id:
        raise HTTPException(404, "Bid packet not found")
    
    return bid_packet.parsed_json


@app.get("/api/v1/bid-packets/{bid_packet_id}/sequences")
async def get_sequences(
    bid_packet_id: str,
    user: User = Depends(get_current_user)
):
    """Get all sequences from a bid packet."""
    sequences = await db.get_sequences(bid_packet_id)
    return {"sequences": sequences}


# ============================================================
# PREFERENCES ENDPOINTS
# ============================================================

class PreferencesInput(BaseModel):
    prefer_weekends_off: bool = False
    preferred_destinations: List[str] = []
    avoid_destinations: List[str] = []
    min_layover_hours: Optional[float] = None
    max_tafb_hours: Optional[float] = None
    maximize_pay: bool = False


@app.post("/api/v1/preferences")
async def save_preferences(
    prefs: PreferencesInput,
    user: User = Depends(get_current_user)
):
    """Save or update user preferences."""
    await db.upsert_preferences(user.id, prefs.dict())
    return {"status": "saved"}


@app.get("/api/v1/preferences")
async def get_preferences(user: User = Depends(get_current_user)):
    """Get current user preferences."""
    prefs = await db.get_preferences(user.id)
    return prefs or {}


# ============================================================
# RECOMMENDATIONS ENDPOINTS
# ============================================================

@app.post("/api/v1/recommend")
async def get_recommendations(
    bid_packet_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get sequence recommendations based on user preferences.
    Returns top 10 sequences ranked by preference match.
    """
    # Get sequences
    sequences = await db.get_sequences(bid_packet_id)
    
    # Get preferences
    prefs = await db.get_preferences(user.id)
    
    # Score and rank
    ranked = score_sequences(sequences, prefs)
    
    return {
        "recommendations": ranked[:10],
        "total_sequences": len(sequences)
    }
```

---

## Rate Limiting

```python
from fastapi import HTTPException
from redis import Redis
from datetime import datetime

redis = Redis()

def check_rate_limit(user_id: str, limit: int = 10, window: int = 86400):
    """
    Limit API calls per user.
    Default: 10 requests per 24 hours.
    """
    key = f"ratelimit:{user_id}:{datetime.now().strftime('%Y%m%d')}"
    
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window)
    
    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {limit} requests per day."
        )
    
    return count
```

---

## LLM Parsing Implementation

```python
import anthropic
import json
from typing import Literal

client = anthropic.Anthropic()

PatternType = Literal["flight_legs", "duty_periods", "layovers", "full_page"]

SCHEMAS = {
    "flight_legs": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "duty_period": {"type": "integer"},
                "date_span": {"type": "string"},
                "equipment_code": {"type": "integer"},
                "flight_number": {"type": "string"},
                "depart_station": {"type": "string"},
                "depart_time_local": {"type": "string"},
                "depart_time_base": {"type": "string"},
                "meal": {"type": "string", "nullable": True},
                "arrive_station": {"type": "string"},
                "arrive_time_local": {"type": "string"},
                "arrive_time_base": {"type": "string"},
                "block_time": {"type": "string"}
            }
        }
    },
    "duty_periods": {
        # ... schema definition
    },
    "layovers": {
        # ... schema definition
    }
}

async def llm_parse(text: str, pattern_type: PatternType) -> dict:
    """
    Parse complex patterns using LLM.
    Uses caching to minimize API calls.
    """
    # Check cache first
    text_hash = hashlib.md5(f"{text}:{pattern_type}".encode()).hexdigest()
    cached = await get_llm_cache(text_hash)
    if cached:
        return cached
    
    # Build prompt
    schema = SCHEMAS.get(pattern_type, {})
    prompt = f"""Extract {pattern_type} from this bid packet text.

Return valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Rules:
- Times are in HHMM format (e.g., "0800", "1430")
- Block times are in H.MM format (e.g., "3.10", "15.45")
- If a value is unclear or corrupted, make your best guess based on context
- Station codes are 3 letters (e.g., "DFW", "ICN", "NRT")

Bid packet text:
{text}

Return ONLY valid JSON, no explanation."""

    # Call LLM
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    result = json.loads(response.content[0].text)
    
    # Cache result
    await save_llm_cache(text_hash, result, response.usage.input_tokens + response.usage.output_tokens)
    
    return result
```

---

## Validation Layer (Pydantic)

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date

class TimeLocal(BaseModel):
    local: str = Field(..., pattern=r"^\d{4}$")
    base: str = Field(..., pattern=r"^\d{4}$")

class FlightLeg(BaseModel):
    leg_type: str = "FLIGHT"
    equipment_code: int
    flight_number: str
    depart_station: str = Field(..., min_length=3, max_length=3)
    depart_time: TimeLocal
    meal: Optional[str] = None
    arrive_station: str = Field(..., min_length=3, max_length=3)
    arrive_time: TimeLocal
    block_time: str
    
    @validator('block_time')
    def validate_block_time(cls, v):
        # Normalize corrupted values
        v = v.replace(':', '.')
        if '.' not in v and len(v) >= 3:
            v = v[:-2] + '.' + v[-2:]
        return v

class DutyPeriod(BaseModel):
    duty_index: int = Field(..., ge=1)
    date_span: str
    legs: List[FlightLeg]
    release_time: TimeLocal
    layover: Optional[dict] = None

class SequenceTotals(BaseModel):
    total_block: str
    total_synthetic: str
    total_tpay: str
    tafb: str

class Sequence(BaseModel):
    seq_num: int
    ops_count: int = Field(..., ge=1)
    positions: List[str]
    report_time: TimeLocal
    calendar_start_dates: List[int]
    duty_periods: List[DutyPeriod]
    ttl: SequenceTotals
    
    @validator('calendar_start_dates')
    def validate_ops_count(cls, v, values):
        if 'ops_count' in values and len(v) != values['ops_count']:
            raise ValueError(f"calendar_start_dates count must equal ops_count")
        return v

class BidPacketPage(BaseModel):
    page_number: int
    base: str
    equipment: str
    division: str
    bidding_period_start: str
    bidding_period_end: str
    sequences: List[Sequence]
```

---

## Project Structure

```
pbs-optimizer/
├── src/
│   ├── __init__.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── ocr.py              # PyMuPDF + Tesseract
│   │   ├── regex_patterns.py   # All regex patterns
│   │   ├── llm_parser.py       # LLM parsing logic
│   │   ├── normalizers.py      # Data cleaning functions
│   │   └── pipeline.py         # Main parsing orchestration
│   ├── expander/
│   │   ├── __init__.py
│   │   └── expand_instances.py
│   ├── scorer/
│   │   ├── __init__.py
│   │   ├── preferences.py
│   │   └── ranking.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app
│   │   ├── routes/
│   │   │   ├── parse.py
│   │   │   ├── preferences.py
│   │   │   └── recommend.py
│   │   ├── dependencies.py     # Auth, rate limiting
│   │   └── models.py           # Pydantic request/response models
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py         # Connection setup
│   │   ├── models.py           # SQLAlchemy models
│   │   └── queries.py          # Database operations
│   └── cache/
│       ├── __init__.py
│       └── redis_cache.py
├── schemas/
│   ├── bid_packet_page_format.json
│   └── example_page_902.json
├── data/
│   ├── raw/
│   ├── parsed/
│   └── samples/
├── tests/
│   ├── __init__.py
│   ├── test_regex.py
│   ├── test_llm_parser.py
│   ├── test_pipeline.py
│   └── test_api.py
├── migrations/                  # Alembic migrations
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Environment Variables

```bash
# .env.example

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/pbs_optimizer

# Redis
REDIS_URL=redis://localhost:6379/0

# Auth
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=86400

# Feature Flags
USE_LLM_PARSING=true
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
```

---

## Development Roadmap

### Week 1: Finish Parser
- [x] OCR extraction (PyMuPDF + Tesseract)
- [x] Regex: SEQ headers
- [x] Regex: RPT times
- [x] Regex: RLS times
- [x] Regex: TTL totals + normalization
- [ ] LLM: Flight legs
- [ ] LLM: Duty periods
- [ ] LLM: Layovers
- [ ] Assemble into complete JSON
- [ ] Pydantic validation

### Week 2: Build API
- [ ] FastAPI setup
- [ ] PostgreSQL + SQLAlchemy
- [ ] Redis caching
- [ ] Parse endpoint
- [ ] Sequences endpoint
- [ ] Preferences endpoint
- [ ] Recommendations endpoint

### Week 3: Basic Frontend
- [ ] Next.js setup
- [ ] Upload page
- [ ] Parsed sequences display
- [ ] Preferences form
- [ ] Recommendations view

### Week 4: Polish & Deploy
- [ ] Auth (Clerk/Auth0)
- [ ] Deploy backend (Railway/Render)
- [ ] Deploy frontend (Vercel)
- [ ] 5 pilot beta testers
- [ ] Iterate on feedback

---

## Next Implementation Step

Add LLM parsing for flight legs. This is the most complex pattern and will validate the hybrid approach works end-to-end.
