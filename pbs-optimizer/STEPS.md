# David Schallipp
# PBS Optimization Engine
# Initial Phase: PDF Parsing for Production ready deployment
# 13 December 2025 

# Leveraging the optimal python libraries/packages to perform high-level parsing and extraction of bid packet data 

Options: 

1. LlamaParse (by LlamaIndex) from Hugging Face + VLM models 
2. Unstructured.io 
2. PyMuPDF (fitz), Docling (open-source, IBM Research) library 
3. Camelot, Tabula-py
4. pdfplumber

Decision (free no API -> open-source): 

Text extractor library (PyMuPDF + deep learning module -> Docling/Camelot)

Innerworkings: 

PyMUPDF handles the raw extraction and then add regex/parsing logic to properly structure it. 

Phase 0.1 will include commencing with PyMUPDF. 

Docs provided = bid packet pdfs, target schema, expected output. 
From here, and acquired understanding is essential: then generalization, and then gauge assessment of column positions and patterns. 

--------------------------------------------------------------------------------------------

Foundation for the Aviation PBS Optimization Engine is the **Data Model**

# Pipeline

Raw Bid Packet (PDF/text) 
    → Parse to JSON (bid_packet_page_format schema)
    → Expand to instances (expanded_for_each_trip_instance)
    → Feed to optimizer (schedule selection)
    → Generate bid recommendations

Phase 1.1: Input raw bid packet -> Output -> JSON aligning with bid packet format schema 

Option #1: PyMUPDF python library + regex for additional parsing 

Optimization of this will set the stage for Phase #2 -> **Pilot Preference Modeling** (where AI comes in).

# Preference Categorization 
# Preference Input UI 
# Preference -> Bid Translation (map input human preferences to PBS bid syntax)

Phase #2 (Pilot Preference Modeling) sets the stage for the **Optimization Engine** Phase #3

# All available sequence instances
# Pilot's seniority (what they can reasonably expect to get)
# Pilot's preferences
# Regulatory constraints (max duty hours, rest requirements)

------> In order to generate:

# Optimal bid configuration
# Predicted schedule outcome
# Alternatives if primary preferences denied

Later Phases: 

1. Seniority Modeling 
2. Conflict Detection 
3. Bid File Generation 
4. XAI (EXPLAINABLE AI) -> Explanation Layer 
5. Schedule Visualization 

--------------------------------------------------------------------------------------------

Tech Stack Current (Dec 19, 2025):

Tech Stack:

Python 3.x
PyMuPDF (PDF handling + OCR)
Tesseract (OCR backend)
Pydantic (schema validation) - planned
Regex (text parsing) - next step

Command to run the parser_pipeline.py module:

python -m src.parser_pipeline

--------------------------------------------------------------------------------------------

Infrastructure Cost Breakdown:

OCR:        $0 (Tesseract)
LLM:        $50-100 (Haiku/GPT-4o mini)
Hosting:    $20-50 (small VPS)
Database:   $0-20 (PostgreSQL)
─────────────────────────────
Total:      $70-170/month

--------------------------------------------------------------------------------------------

Approach: Hybrid 

Strategy:

Use PyMuPDF + Tesseract for text extraction (free)
Use regex for simple, reliable patterns (SEQ, RPT, RLS)
Use LLM for complex patterns (flight legs, duty periods) or as fallback when regex fails
Use Pydantic for validation.

Define DB schema and database that will be used. 

--------------------------------------------------------------------------------------------

llm_client.py - Handles LLM interactions

create_client() - Creates OpenAI connection
parse_flight_legs_with_llm() - Uses LLM to parse flight legs

parser_pipeline.py - Main parsing logic

extract_text_with_ocr() - Gets text from PDF
parse_seq_headers() - Regex for SEQ headers
parse_rpt_times() - Regex for RPT times
parse_rls_times() - Regex for RLS times
parse_ttl_totals() - Regex for TTL totals
parse_bid_packet() - Orchestrates everything
Imports parse_flight_legs_with_llm from llm_client.py

--------------------------------------------------------------------------------------------

22 December 2025 

5 new modular files created: 

1. models/schemas.py = defines the data structure 
2. llm/client.py = Talks to OpenAI API 
3. ocr.py = Attemps to extract text from the PDF using OCR (optical charcater recognition, Deep Learning)
4. validators.py = Cross-checks against LLM output 
5. config.py = configuration settings 

6. Next (engine.py) = the orchestrator that ties everything together 

NOTES: 

- Regex is not for parsing anymore, it's for validation 
- OCR -> LLM (does the parsing) -> Regex (checks the work)

CONSIERATION: 

HITL Feedback System: 

Why? edge cases exist. Pilots need to trust system. Continuous improvement -> human corrections become training data. 

NOTE: 

OCR is necessary for LLM. Does: image -> text.

Parsing = text -> structured JSON.

**Flow:**

1. Receive file_path
2. Attempt to extract text directly
3. If empty → call OCR
4. If still empty → HALT, return error "Could not extract text"
5. Send text to LLM for parsing
6. If LLM returns invalid JSON → RETRY once
7. If retry fails → HALT, return error "LLM parsing failed"
8. Check structure (does 'sequences' key exist?)
9. If wrong structure → HALT, return error "Invalid response structure"
10. Cross-check with regex (sequence count, TTL count)
11. If mismatch → set requires_review = True, add warning
12. Validate with Pydantic
13. If validation fails → set requires_review = True, add errors
14. Return result with data, errors, warnings, requires_review flag

Orchestration Flow = (OCR → LLM → Regex Cross-check → Schema Validation)

CONSIDERATIONS FOR PRODUCTION HANDOFF: 

Cost Tracking: Since we are using an LLM (GPT-4o), every call costs money. A production engine should log how many "tokens" were used per PDF so we can calculate your profit margins accurately.

Telemetry: Add a log.info(f"Parsed {file_path} in {time_elapsed}s"). If the OCR takes 30 seconds, a pilot will get bored and close the app. You need to measure this.

**COST EVAL:** 

GPT-4o-mini pricing:

Input: $0.15 per 1M tokens
Output: $0.60 per 1M tokens

A typical bid packet page might use:

~3,000 input tokens (OCR text + prompt)
~2,000 output tokens (JSON response)
Cost: ~$0.002 per page

If a pilot uploads 50 pages: ~$0.10 per bid packet
At $50/month subscription, that's 500:1 margin.

**Note**

Non-determinism in LLMs. 

**Value Proposition** 

The engine that orchestrates, validates, and flags uncertainty. 

**Solutions to LLM non-determinism**

1. Set temperature to 0. 
2. Add retry on mistmatch logic. If regex count ≠ LLM count, retry automatically
3. Track success rate metrics. Log each runs accuracy to check improvements over time. 
With each run count the amount of times the metrics output to 0 in order to improve the system performance. 

**File Structure Reference Guide:**

src/
├── llm/client.py       ← Prompts live here (add temperature=0)
├── engine.py           ← Orchestrator
├── models/schemas.py   ← Pydantic validation
├── validators.py       ← Regex cross-checks
├── ocr.py              ← PDF extraction
└── config.py           ← Settings

**Commands to run engine**

python engine.py --save 
src (root)
ensure environement is activated
python -m engine.py

**LLM considerations:**

gpt-4o-mini
Anthropic contains higher cost eval and higher token usage consumption.

**revisit this section**

**Hybrid Validation Architecture**

The system doesn't just use a standalone LLM, due to the risks of AI. 
The 'double check pattern': The system uses determinstic Regex Parsers to establish a 'ground truth' thereby counting total sequences and financial totals before the LLM attempt. 

Conflict detection: If the LLM output does not match the regex counts, the system automatically flags the record with requires_review: True and lowers confidence_score. This prevents silent failures which are the biggest risks in production AI systems. 

Schema enforcement: The use of pydnatic over standard Python dictionaries or dataclasses. 
Every LLM response is enforced through a strict Pydantic schema. This ensure that even if the LLM hallucinates a string, the system will catch the type mismatch before it reaches the database. 

If the schema validation fails, the engine captures the specific 'loc' and 'msg' of the error, allowing for targeting retries or developer debugging. 

Strong recommendation for **parallel processing/chunking** in Phase 2.

**Very important to look into 'edge cases'**

Path (A) Can we extract the data successfully? ->
Path (B) Can we make the extraction reliable and optimize it?

--------------------------------------------------------------------------------------------

Tuesday, December 23, 2025 

Next Steps: 

1. Implement "Self-Correction" Loops (The Retry Logic)
2. Transition to "Chunked Parsing"
3. Establish a "Ground Truth" Evaluation Suite
4. Harden the Infrastructure (Logging & Observability)
5. Build the "Instance Expander" (The Business Logic)

Consider a "Spatial Reconstruction" Algorithm - a custom algorithm that uses OCR coordinates to "reconstruct" that grid into a structured sequence before the LLM sees it.

"Validation Logic" = "Dual-Track Verification Engine."

--------------------------------------------------------------------------------------------

Ground Truth Comparison 

In ML/AI systems, you need to track whether your system is truly accurate, or just confident. 

Ground truth = the known correct answer 
Prediction = what the system produces (engine's output)

Through comparing these two points, we obtain **accuracy metrics** in order to objectively measure if system works well. 

--------------------------------------------------------------------------------------------

December 26, 2025 

src/llm/client.py

Need to update the prompts to address 3 issues: 

1. Deadheads not recognized
2. Meals missing (L/D/B)
3. Duty periods merged

REMAINING ISSUES: 

calendar_start_dates27OCR can't read calendar grid
date_span_md wrong
meal codes
Deadhead not recognized
Minor totals

New Additions: 

Added Will's full length pdf packet to backend docs and then tested OCR engine on it. 
The OCR performed well and was able to output the correct page count, and extract visible structure. 

**Very Important for when we build the UI**

Option 2: API Endpoint (Good for Production)
python# FastAPI endpoint
@app.post("/parse")
async def parse_bid_packet(file: UploadFile):
    # Save uploaded file
    # Parse it
    # Return results

(for uploading bid packets so it doesn't have to be manually changed (the route) in config.py)

--------------------------------------------------------------------------------------------

December 27, 2025 

Page 7 Packet -> Parsing Results:

Example PDF was extracted from this full packet
Parser works on the real production data
Calendar dates remain the main accuracy issue

Also time processing is an issue -> we need parallel processing for production.

Multi-page parsing -> working

Schema validation -> working

Cost tracking -> working 

Calendar dates -> broken (OCR issue)

Production speed -> too slow (needs parallelization)

**Important Next Steps**

1. Fix calendar dates first (Spatial-Regex Algorithm)
2. Then add caching (parse once per bid packet)
3. Then add parallel processing (if needed)

**Interview Talking Point:**

"I built a deterministic validation layer that cross-references AI outputs against document metadata to eliminate hallucination in structured extraction."

**Spatial-Regex Algorithm:**

Since Tesseract and other OCR tools struggle to keep the numbers (1–31) aligned with their headers (MO, TU, etc.)

The "Grid Anchor" Algorithm
Instead of looking at the text as a flat string, the algorithm treats the right side of the page as a coordinate-based zone.

1. Zone Isolation
We use the headers "MO TU WE TH FR SA SU" as a horizontal "anchor." Everything directly below that header, extending to the right margin, is defined as the Calendar Zone.

2. The Regex "Sieve"
Within that isolated zone, we don't care about the dashes (--) or the messy OCR characters. We run a regex pattern specifically for integers between 1 and 31.

The Logic: \b([1-9]|[12][0-9]|3[01])\b

The Capture: This pulls out a clean list of numbers (e.g., [7, 11]).

3. Cross-Validation (The "Ops Count" Check)
This is the most important part of the algorithm. We know from the sequence header how many times the trip runs (e.g., 2 OPS).

If the Regex finds 2 numbers, the confidence is High.

If the Regex finds 3 numbers but the header says 2 OPS, the algorithm flags a mismatch.

Developed the algorithm. Now need to properly place into the pipeline/engine.

Current **engine.py** flow:

PDF → OCR → LLM → **Calendar Override** → Validate → Return

MVP Parser = **Complete** (Can optimize later).

Next Priorities (Sequentially):

1. Caching - Parse once, store results (eliminates repeat processing)
2. Batch processing - Parse all 965 pages, store to database
3. API layer - FastAPI endpoint for frontend
4. Parallel processing - Reduce 21 hours → 2 hours (if needed)

**Phase 1: Data Pipeline Foundation (Parser Pipeline) Summary:**

End-to-end system that:

1. Ingests messy, multi-page PDFs.

2. Extracts complex data using a Hybrid AI/Regex approach.

3. Validates its own work and flags errors for humans.

4. Tracks its own costs and performance metrics.

**Add automated testing feature** (pytest)

Phases of Deployment/Development:

Exploration -> Stabilization -> Production

Types of automated testing:

1. Unit tests (tests individual functions)
2. Integration tests (test module interactions)
3. Regression tests 

pip install pytest 
cd src 
pytest tests/ -v

--------------------------------------------------------------------------------------------

Dec 29, 2025: **Immediate Next Steps:**

Add automated tests for calendar_resolver

Fix GROUND_BETWEEN_LEGS parsing

Set up database + caching

Build batch/parallel processor for all pages

(A) Automated tests 

Automated Testing allows for: 

1. Prevent regressions - If you change something later, tests catch if you broke it
2. Document behavior - Tests show what the code should do
3. Confidence to refactor - Change code fearlessly knowing tests will catch errors
4. Prove it works - Not "I think it works" but "I can demonstrate it works"

pytest is a testing framework that:

Discovers files named test_*.py
Finds functions/methods starting with test_
Runs each one and reports pass/fail 

'Arrange, Act, Assert.'

In order to run the test run: 

pytest -v (file_name)

Test complete for calendar_resolver algorithm. 

Now need to implement testing for engine.py orchestration module. 

**engine.py** does the following:

Calls OCR to extract text from PDF
Calls LLM to parse the text
Cross-checks with regex validators
Overrides calendar dates with resolver
Validates with Pydantic schema
Returns ParseResult with metrics

--------------------------------------------------------------------------------------------

Dec 31, 2025: Database Exploration and Set up

Tech: Supbase

Why databases: Current system is stateless (no memory of users, preferences, history)

**Database enables:**

FIRST TIME:
PDF → Parse → Store in database → Return JSON

EVERY TIME AFTER:
Request → Fetch from database → Return JSON (instant, free)

Need to keep track of several characteristics (Users, preferences, blocked dates, bid history, outcomes).

Will's Docs: Bidding Workflow: 


1. Review Dashboard (seniority, LCW, existing credit)
2. Search/View Pairings
3. Create Pairing Pools per layer
4. Add Line Construction Preferences
5. Validate bid in Layer Tab
6. Submit before deadline
7. Await award

Components need to be built from database: 

1. User preferences storage
2. Layer-based bidding logic
3. Seniority-aware ranking
4. Reserve vs Lineholder logic

Value Prop: 

Parse the bid packet → Show all available pairings
Remember preferences → "You like Fridays off and Hawaii trips"
Score pairings → "SEQ 227 matches 8 of your 10 preferences"
Rank for them → "Here's your optimal bid list for each layer"
Track outcomes → "Last month you got your #3 choice"

Pilot types: "I want Fridays off, short trips, and Hawaii layovers. 
             My kid has soccer on Tuesdays so I need to be home by 6pm."
        ↓
Your system translates to structured preferences:
  - days_off: [Friday]
  - max_trip_length: 3
  - preferred_layovers: [HNL, OGG, LIH]
  - arrive_before: 18:00 on Tuesdays
        ↓
Your system scores EVERY sequence against these preferences
        ↓
Output: "Here are your top 50 sequences, ranked. 
        SEQ 227 scores 94/100 - Hawaii overnight, home Thursday, 2-day trip."
        ↓
Pilot takes this ranked list and enters it into PBS

**Need to build The Natural Language Preference Parser**

Sequential Dependency: 

   Database Schema                                            
│        ↓                                                     │
│   NL Preference Parser (outputs to schema)                   │
│        ↓                                                     │
│   Scoring Engine (reads schema, compares to sequences)       │
│        ↓                                                     │
│   API (serves scored results)                                │
│        ↓                                                     │
│   Frontend (displays to user)         

--------------------------------------------------------------------------------------------

Jan 6, 2025: Database Schema Design:

- Database schema is designed to handle the complex, multi-layered data found in airline bid packets.
- It uses a hierarchical structure to organize data from the broad page level down to individual flight details.

Layers:

1. Page Metadata Layer: root of the schema. Captures the global context for every flight on a specific page, such as the base, equipment, and bidding period dates. 
2. Sequence Layer: A sequence is a template for a trip. Attributes -> includes the seq. number of times it operates (OPS) and crew positions
3. Duty Period Layer: date spans and release times that capture exactly when the pilot is off the clock
4. Legs and Layover Layer: flight legs (detailed records for every takeoff and landing), ground/deadhead, layovers

The schema concludes with a 'TTL (Total) Section.' This aggregates the math for the entire sequence: 
- Total pay hours
- Total time away from base 
- Block time 

This is what the scoring engine will use to rank trips according to pilot preferences. 
The **Scoring Engine** is the core bussiness logic that ranks thousnads of available trip sequences based on a pilot's specific personal preferences. 

The **parser** focuses on extracting data from the PDF, the scoring engine focuses on evaluating that data. 

How the **Scoring engine works**:

It operates as a mathematical 'weighing' system. It takes the structured JSON data produced by the parser and applies numerical weights to various attributes of a trip. 

Input = pilot preferences ..... ------> Scoring loop (engine loops through every trip instances and calculates a "Total Score" for each one): 

**Scoring Metrics:** 
Base score = starts at 0 
Date Match = if != pilots preference = -100 points 
Destination Match = == pilots preference = +50 points 
Pay Match = Multiply Total TPAY hours by a 'Pay Weight' to output a score 
Timing Match: If != pilot's preference -> -20 points 

-------> Output: Ranked Recommendations: 

Once every trip has a score, the engine sorts them from highest to lowest. The pilot sees that curated list.

The **database is crucial to feed into the input layer** as the scoring engine needs the structured JSON data was parsed from the bid packets and stored in the database 

Logic Layer: 

In a backend module: iterates through every possible trip instance and uses a weighted algorithm to calculate a numerical score for each sequence. 
Hard Constraints = filters out trips that are physically impossible or illegal for the pilot to fly **(FAA rest requirements)**
Soft Constraints: applies the user's weights to various attributes like Total TAFB or specific arrival stations. 

Output Layer = once every trip is scored, the engine sorts the list from highest to lowest. 
The FastAPI backend then sends this ranked list to the frontend, allowing the pilot to see their top 10 recommendations.

This **scoring engine** entails translating a pilot's personal life into a mathematical model. 

At the heart of the engine is a weighted scoring algorithm. In software architecture, this is often called a 
**Weighted Decision Matrix**

Flow: 

PDF → Parser → Parsed JSON → Stored in sequences table
                                    ↓
User preferences → Stored in preferences table
                                    ↓
Scoring Engine reads both → Produces ranked list
                                    ↓
User submits bid → Stored in user_bids table
                                    ↓
Outcome reported → Stored in bid_outcomes table (flywheel)

**Flywheel Architecture:**

Designed to solve the problem of AI non-determinism (variability) in pilot scheduling. 
The flywheel is built on the interaction between Scoring Engine and real-world data:
Data Extraction: Your hybrid system uses Deterministic Anchors 
(Regex) to count sequences and Probabilistic Agents (LLMs) to parse complex details.

Validation: The engine cross-checks these two sources. 
If the counts don't match, it triggers a Self-Correction Loop.

Accuracy Improvement: Every time the system self-corrects or a pilot provides HITL (Human-in-the-Loop) feedback, 
the data becomes cleaner.

Distribution Moat: As accuracy hits 100% (which spatial reconstruction algorithm aims for), 
more pilots use the tool, providing more "ground truth" data.

Compounding Value: More data allows for better Seniority Modeling, making the tool even more indispensable 
for the 16,000 pilots at American Airlines.

--------------------------------------------------------------------------------------------

Jan 8, 2025 

'Supabase' database implementation and design 

Tasks: 

Create Supabase project
Create 8 tables via dashboard or SQL
Set up Row Level Security policies
Install supabase-py in backend 
Write Python functions to insert/query
Store first parsed bid packet

Database tables:

users, preferences, blocked_dates, preference_snapshots, bid_packets, sequences, user_bids, bid_outcomes

--------------------------------------------------------------------------------------------

Jan 13, 2025 

Creating the Supabase project

Option: email sign up

Information entered manually: 

Seniority, bid status, base, equipment

No employee number: Avoids "nonpublic company information" concern

**users table**

email (supabase auth)
name 
base 
equipment 
position 
seniority_number 
division 
employee_number

**Important implication**

You can't verify they're actually a pilot. Anyone could sign up and claim to be a DFW 777 Captain.

To scale publicly:

Invite codes, manual approval, and some additional verification step 

updated schema change: 

employee_number VARCHAR(20) NULLABLE

Bid Packets Table:

"This table acts as the storage layer for the OCR and LLM parsing pipeline you've built. 
When a PDF is processed by your engine.py, the metadata (like the equipment type and base) is stored here, 
while the file_hash ensures you can leverage the caching strategy we discussed to save on LLM costs."

ML handoff: 

the preference_snapshots and bid_outcomes tables are perfect for training recommendation models 
and tracking the 'flywheel' effect of pilot choices

**Core Components Section**

1. PDF Parser 
2. Calendar Resolver Algorithm 
3. Automated tests
4. Database schema 
5. Authentication 
6. Auto-generated API

--------------------------------------------------------------------------------------------

Jan 14, 2025 

Algorithm tracing and preparation: 

**Next Path (Phases)**

1. connect parser to database 
2. build NL preference parser 
3. build scoring engine 

Path 1:

Install supabase-py in project
Write function to convert parser output → database insert
Parse a bid packet and store it in sequences table
Verify data is queryable

Path 2: 

LLM-based system that converts user preferences into structured preferences that gets stored in preference table 

Path 3:

The core business logic that:

Reads user preferences from database
Reads sequences from database
Scores each sequence against preferences
Returns ranked list

--------------------------------------------------------------------------------------------

Tuesday, January 20, 2025: 

Files Revisited:

engine.py           ← Main parser orchestrator
calendar_resolver.py ← 100% accurate date extraction
llm/                ← LLM client
models/             ← Pydantic schemas
parser/             ← Parser modules
scorer/             ← Scoring engine (started?)

**Command to test parsing pipeline engine on a page:**

python engine.py --page 7

Preparation for parser output to database mapping:

Tables: 

bid_packets
sequences

conversion functions needed:
1. Time String to Minutes: "H.MM" format
2. Clock Time to Minutes: "HHMM" format
3. Extract Layover Cities
4. Division Mapping
5. Derive Pairing Length
6. Derive is_redeye
7. Derive is_transoceanic

----> Parser Output (JSON) -----> Transformation ------> Supabase Tables

Created: 

src/db/supabase_client.py - Connection setup

created .env file
(SUPABASE_URL) and (SUPABASE_KEY)
installed python dotenv 

--------------------------------------------------------------------------------------------

Wednesday, January 21, 2025 

Building the store packet function, creating the transformation utilities 

created 'transformers.py' module in order to transform some parser output to database ready format

Logic was largely done by function-based logic and object-oriented programming 
+ Mapping keys and dictionaries 

Furthermore, and very importantly, we created a cryptographic hash function (SHA-256)
meaning that any change to the file produces a compeltely different hash: 
Serves as a unique identifier for a bid packet PDF file 
Allows for: 
1. Deduplication
2. Change detection 
3. Data Integrity 

**Need to fix and revisit def_is_transoceanic_trip function later**

Future fix (V2):

Use a lookup table in your database
Or use a geolocation library that maps IATA codes to continents
Or derive from flight time (>8 hours = likely transoceanic)

command to test file: python -m src.db.transformers

Next Step (Now): The store packet function 

parser output --> database 

store_packet.py must:
│  1. Read JSON       │
│  2. Check for dupe  │
│  3. Insert bid_packet│
│  4. Transform each  │
│     sequence        │
│  5. Insert sequences

Overall purpose of store_packet.py is to take parser output and store it in the database 

Purpose of storing engine output to a database is:

Data persists (doesn't disappear)
It's queryable (find all Hawaii trips)
The scoring engine can read it
Multiple users can access it

**Understanding the Engine:**

Parser returns a ParseResult object with:

@dataclass
class ParseResult:
    success: bool
    data: dict              # The sequences
    errors: list
    warnings: list
    requires_review: bool
    confidence: str         # "high", "medium", "low"

Now we are creating a pipeline to perform the following:

Parser → ParseResult object → Store in DB
              ↓
         Metrics preserved (success, confidence, errors)

From here we are building a massive orchestration file **(pipeline.py)** that
orchestrates everything together.

Pseudocode:

def process_and_store_page(pdf_path, page_num)
// parse a page and store results in database
// parse page using existing engine 
parse_result = parse_page(pdf_path, page_num)
// quality gate
if not parse_result.success: 
    return reason
// store in db with quality metrics 
storage_result = store_parsed_packet(pdf_path, parse_result)
return the storage result

Full logic (pipeline):

def process_full_packet(pdf_path: str) -> dict:
    """
    Parse and store an entire bid packet.
    """
    
    # Get total pages
    total_pages = get_pdf_page_count(pdf_path)  # 972
    
    # Create bid_packet record once
    packet_id = create_bid_packet(pdf_path, total_pages)
    
    # Parse each page
    for page_num in range(total_pages):
        result = parse_page(pdf_path, page_num)
        
        if result.success:
            store_sequences(packet_id, result.data, source_page=page_num)
        else:
            log_failed_page(packet_id, page_num, result.errors)
    
    # Mark complete
    finalize_packet(packet_id)
    
    return {"packet_id": packet_id, "pages_parsed": total_pages}

**Concern Flag**

Revisit this later: 

    # Load test data
    # Revisit this later 
    test_json = Path(__file__).parent.parent.parent / "data" / "parsed" / "engine_output.json"
    test_pdf = Path(__file__).parent.parent.parent / "data" / "raw" / "DFW_JAN.pdf"

Very Important: **A pilot should be able to:**

1. Sign up / Log in                    ← Supabase Auth (built-in, ~1 hour to wire up)
2. Upload or select bid packet         ← Need full packet processing
3. Enter preferences in plain English  ← Need NL Preference Parser
4. Get ranked sequences                ← Need Scoring Engine
5. Save/export their bid               ← Need API endpoints

Next Steps (Important):

**(src/db/process_full_packet.py)**
We will need to eventually proceed with parsing the full packet (all pages) to store in the database
in order to visualize and demo performance. However, we can skip the step for now, but must revisit later. 

This is what the logic will look like: 

# src/db/process_full_packet.py

def process_full_packet(pdf_path: str, start_page: int = 0) -> dict:
    """
    Parse and store all pages from a bid packet.
    
    - Tracks progress in database
    - Can resume from where it left off
    - Logs results for each page
    """

The next two important features are the two algorithms that must be developed which encompas the core bussines logic. 

That is: 
1. NL Preference Scorer Engine (core differentiator)
2. Scoring engine (core product logic)

First we will perform psuedocode and algorithm tracing for our first algorithm to adhere to engineering best practices.

**Creation of Core IP for Company**

2 algorithms -> NL Preference Scorer Engine & Scoring Engine

**Constraints:**
- Must handle ambiguous language
- Must map to valid PBS preference types (from the 50+ we extracted)
- Must handle conflicting preferences gracefully
- Must be explainable (pilot understands what was extracted)

Edge Cases Algorithm Must Handle:

Edge CaseHandled?HowAmbiguous language✓Asks clarifying questionsContradictory preferences✓Detects and explains conflicts
Implicit context✓
Infers with confirmationTypos/informal language✓Preprocessing + fuzzy matchingEmpty input✓
Offers guided questionsExpert technical input✓
Preserves precisionMissing information✓Assumes with explicit confirmation

Core **Must** components of algorithm:

reprocessing -> LLM Extraction ----> ****VALIDATION*** very important and good 
----> to normalization ----> conflict detection ---> confidence scoring ----> final output

Next Steps
Now that we've validated the design with edge cases and external research:

Finalize the preference schema (add minimize_credit, buddy_bid)
Design the LLM prompt for extraction
Build the validation rules (the core IP)
Implement and test

Developing code file and stress-tested to verify accuracy logic 
Code file title: src/preferences/schema.py
**About: Preference Schema Model**
**Overview**
src/preferences/schema.py is the domain model for pilot scheduling preferences. It defines the complete data structure 
for representing what pilots want in their schedules, provides validation logic to ensure data integrity, 
and includes utilities for serialization and natural language parsing support.

**Purpose:** Serves as the single source of truth for all preference types, their valid values, and their relationships.

Docstring for pbs-optimizer.src.preferences.schema
single source of truth for preference types
contains all validation logic in one place
heavily documented, and includes tests

Capabilities:

71 preference fields across 8 categories
Type and range validation
Conflict detection (same city in prefer/avoid)
Cross-category validation (turns + layover = impossible)
JSON serialization/deserialization
Human-readable summaries
Synonym mappings ready for NL parsing

**Next Step: LLM Prompt Design (Layer 2 (LLM Extraction) of the NL Parser).**

The prompt will:

1. Take natural language input
2. Extract structured preferences
3. Map to our schema

--------------------------------------------------------------------------------------------

Friday, Jan 23, 2025

Latency, compute, memory, and efficiency 

5. Design the LLM prompt ← Next step
6. Implement the parser
7. Test with real inputs

--------------------------------------------------------------------------------------------

Thursday, Jan 29, 2025 

Setting up prompt for NL preference parsing logic. 

File structure:

src/preferences/
├── __init__.py
├── schema.py              ← Done (validation, types)
├── prompt_template.py     ← Creating now (the prompt)
├── examples.py            ← Creating now (few-shot examples)
└── parser.py              ← Creating after (API call + orchestration)

Completed: 

1. PDF Parser (src/engine.py)
2. Database Schema (supabase)
3. Parser -> DB Pipeline (src/db/store_packet.py)
4. Data Fransformers (src/db/transformers.py)
5. Preference Schema (src/preferences/schema.py)

We finished the Preference Schema: 
the complete data model for pilot preferences (71 fields, validation, conflict detection, tests passing).

**Now designing the LLM prompt**

NL Preference Parser is done.

How it works: 

User input: "No redeyes, commuter from Denver, home by 6pm" 

----> 6 Layer Pipeline 

1. Preprocessing 
2. LLM Extraction (Claude API)
3. Validation 
4. Normalization 
5. Conflict Detection 
6. Confidence Scoring 

Output: PilotPreferences object + confidence + warnings

Files & Roles:

src/preferences/
├── schema.py           ← Defines WHAT preferences can exist
├── examples.py         ← Teaches the LLM HOW to extract
├── prompt_template.py  ← Builds the message sent to Claude
└── parser.py           ← Orchestrates the entire pipeline

The Data Model -> Defines all valid preference types and how to validate them
(Enums constraints, dataclasses for structure, validation functions, synonym mappings, pilotpreferences)
The schema enforces structure for the LLM 

Prompt Template: 

┌─────────────────────────────────────────────────────────┐
│  PART 1: SYSTEM INSTRUCTIONS                            │
│  "You are a preference extraction system..."            │
│  "Output ONLY valid JSON..."                            │
│  "Expand synonyms like Hawaii → [HNL, OGG...]"         │
├─────────────────────────────────────────────────────────┤
│  PART 2: SCHEMA DEFINITION                              │
│  "days_off.days_off: List[Day] where Day = Mon|Tue..." │
│  "avoid_pairing_type: List[Type] where Type = ..."     │
│  (Compressed format to save tokens)                     │
├─────────────────────────────────────────────────────────┤
│  PART 3: FEW-SHOT EXAMPLES                              │
│  "Input: 'Fridays off' → Output: {days_off: [Friday]}" │
│  (12 examples from examples.py)                         │
├─────────────────────────────────────────────────────────┤
│  PART 4: USER INPUT                                     │
│  "Extract preferences from: 'No redeyes, home by 6pm'" │
└─────────────────────────────────────────────────────────┘

Up next -> Algorithm 2: Scoring Engine 

**Scoring Engine**

Weighted scoring and matching to preferential logic to provide guidance and feedback to users (pilots).

Current Coverage (V1):

TODO: Scoring Engine V2 - Complete Field Coverage
=================================================
Priority: Post-MVP
Estimated Hours: 4-6

Fields Not Yet Scored (46 remaining):

Days Off:
  - maximize_total_days_off
  - maximize_block_days_off
  - min_consecutive_days_off
  - prefer_days_off_start_day

Pairing:
  - max_duty_periods
  - max_legs_per_duty
  - max_tafb_credit_ratio
  - min_credit_per_duty_minutes

Times:
  - mid_pairing_report_after_minutes
  - mid_pairing_release_before_minutes

Layovers:
  - min_layover_hours
  - max_layover_hours
  - prefer_landing_city
  - avoid_landing_city

Credit:
  - target_credit_min_minutes (partial)
  - target_credit_max_minutes (partial)

Work Blocks (all 8):
  - min_work_block_days
  - max_work_block_days
  - prefer_work_block_days
  - min_days_off_between_blocks
  - prefer_work_block_start_day
  - prefer_pairing_mix
  - allow_reduced_post_block_rest
  - allow_co_terminal_mix

Reserve (all 9):
  - reserve_type_preference
  - avoid_reserve
  - min_reserve_block_days
  - max_reserve_block_days
  - prefer_fly_through_days
  - reserve_must_off_dates
  - reserve_prefer_off_dates
  - waive_reserve_block_of_4_days_off
  - allow_single_reserve_day_off

Misc (all 12):
  - prefer_deadheads
  - avoid_deadheads
  - prefer_deadhead_first_leg
  - prefer_deadhead_last_leg
  - buddy_bid_employee_numbers
  - prefer_one_leg_first_duty
  - prefer_one_leg_last_duty
  - prefer_extended_domicile_rest
  - allow_reduced_domicile_rest
  - allow_double_up
  - allow_same_day_pairing
  - allow_rest_at_outstation

Created:

src/scoring/engine.py is the recommendation engine for pilot scheduling. 
It takes a pilot's preferences and a list of available sequences, then ranks them 
from best to worst match with explainable reasoning.
Purpose: Convert subjective preferences into objective, comparable scores.

Next: API Layer: 

Exposes the backend logic as an HTTP endpoint that the frontend can call. 

--------------------------------------------------------------------------------------------

Monday, February 2nd, 2025 

API Layer = Complete:

POST /api/preferences/parse    ✓ Working
GET  /api/packets              ✓ Working  
GET  /api/sequences            ✓ Working
POST /api/sequences/score      ✓ Working

Next steps: Front-end UI completion 

**Core user flow:**

step 1: enter preferences & analyze preferences
step 2: confirm extracted preferences -> confidence, store sequences 
step 3: view ranked results 
step 4: drill into details 

**Components:**

landing page 
pilot login & auth 
dashboard (scoring)
admin panel 
settings 
stripe payments 

**Recommended stack**

Next.js + Tailwind + shadcn/ui

**1. Next.js project setup**

Next.js installation complete and components installed 

Next step: **landing page**

Landing page = complete:

Components = 

{ } 

running the server = 

npm run dev 

running the MVP locally on another machine = 

brew install ngrok
ngrok http 3000

OR deploying to vercel -> = npx vercel

directory = pbs-frontendnpm 

**Marketing and User Acquisition**

**Components of Landing Page**

page.tsx = This is the entry point. It imports all six sections and renders them in order, 
top to bottom. Nothing else — no logic, no state, no data fetching. It's a pure layout file.

layout.tsx = This wraps every page in the entire app, not just the landing page. 
It does three things: loads the Geist font family, sets the SEO metadata 
(title, description, keywords that search engines read), and applies the font classes to the <body> tag. 
Every route — /login, /dashboard, /pricing — will inherit this layout.

globals.css = This is the design system in a single file. It does two things. 
First, it defines shadcn/ui's CSS variables (all the --background, --foreground, --primary values) 
for both light and dark modes — these power every shadcn component we use. 
Second, it defines your custom BidLine color tokens: --color-navy, --color-amber, --color-slate-warm, etc. 
When we write bg-navy or text-amber in a component, Tailwind resolves those class names through these variables.

--------------------------------------------------------------------------------------------

Date: Tuesday, February 10th, 2026 

Next Phase: 

Dashboard UI (upload → preferences → ranked sequences)

Front-end stack (Current):

Next.js (App Router)
Tailwind CSS
shadcn/ui components
Framer Motion
Geist font

Brand: BidLine (navy + amber palette)

Created:

api.ts:

/**
 * API Client for PBS Optimizer Backend
 * =====================================
 * 
 * This file handles all communication between the Next.js frontend
 * and the FastAPI backend running on localhost:8000.
 * 
 * WHY THIS EXISTS:
 * - Centralizes all API calls in one place
 * - Defines TypeScript types for request/response shapes
 * - Handles errors consistently
 * - Makes it easy to change the API URL later (e.g., for production)
 */

 








































