# Pydantic models for bid packet parsing
# These define the CONTRACT - what valid data looks like

from pydantic import BaseModel, Field, validator
from typing import Optional 

'''
This scehema acts as a contract for validation when the parser extracts data from PDF -> JSON. 
It will validate against these models to ensure correctness before feeding it to the optimizer. 
'''

class TimeLocalBase(BaseModel):
    """Time in both local and base timezone."""
    local: str = Field(..., pattern=r"^\d{4}$")  # HHMM format
    base: str = Field(..., pattern=r"^\d{4}$")


class Departure(BaseModel):
    station: str = Field(..., min_length=3, max_length=3) # Field = constraints
    time_local: str = Field(..., pattern=r"^\d{4}$")
    time_base: str = Field(..., pattern=r"^\d{4}$")


class Arrival(BaseModel):
    station: str = Field(..., min_length=3, max_length=3)
    time_local: str = Field(..., pattern=r"^\d{4}$")
    time_base: str = Field(..., pattern=r"^\d{4}$")


class FlightLeg(BaseModel):
    """A single flight leg within a duty period."""
    leg_type: str = "FLIGHT"
    equipment_code: int
    flight_number: str
    depart: Departure
    meal: Optional[str] = None
    arrive: Arrival
    block_hhmm: str


class GroundLeg(BaseModel):
    """Ground time between flight legs."""
    leg_type: str = "GROUND_BETWEEN_LEGS"
    ground_hhmm: str
    equipment_change: bool = False


class DeadheadLeg(BaseModel):
    """Deadhead (non-working) flight leg."""
    leg_type: str = "DEADHEAD"
    equipment_code: int
    flight_number: str
    depart: Departure
    meal: Optional[str] = None
    arrive: Arrival
    deadhead_carrier_code: str
    deadhead_credit_hhmm: str


class Layover(BaseModel):
    """Layover information between duty periods."""
    city_code: str = Field(..., min_length=3, max_length=3)
    hotel_name: str
    transport_provider: str
    odl_hhmm: str


class DutyTotals(BaseModel):
    """Totals for a single duty period."""
    block_hhmm: str
    synthetic_hhmm: str
    tpay_hhmm: str
    duty_hhmm: str
    fdp_hhmm: str


class DateSpan(BaseModel):
    start: str  # M/D format
    end: str


class DutyPeriod(BaseModel):
    """A duty period within a sequence."""
    duty_index: int = Field(..., ge=1)
    date_span_md: DateSpan
    legs: list  # Can be FlightLeg, GroundLeg, or DeadheadLeg
    release_time: TimeLocalBase
    duty_totals: DutyTotals
    layover: Optional[Layover] = None


class SequenceTotals(BaseModel):
    """Totals for an entire sequence."""
    total_block_hhmm: str
    total_synthetic_hhmm: str
    total_tpay_hhmm: str
    tafb_hhmm: str


class Sequence(BaseModel):
    """A complete sequence block."""
    seq_num: int
    ops_count: int = Field(..., ge=1)
    positions: list[str]
    sequence_label: Optional[str] = None
    sequence_report_time: TimeLocalBase
    calendar_start_dates: list[int]
    duty_periods: list[DutyPeriod]
    ttl: SequenceTotals
    
    @validator('calendar_start_dates')
    def validate_ops_count(cls, v, values):
        if 'ops_count' in values and len(v) != values['ops_count']:
            raise ValueError(f"calendar_start_dates count ({len(v)}) must equal ops_count ({values['ops_count']})")
        return v


class BidStatus(BaseModel):
    base: str
    equipment_family: str
    division: str


class BiddingPeriod(BaseModel):
    start_md: str
    end_md: str
    length_days: int


class PageMetadata(BaseModel):
    page_number: int
    issued_date: str
    effective_date: str
    bid_status: BidStatus
    bidding_period: BiddingPeriod


class BidPacketPage(BaseModel):
    """Complete parsed bid packet page."""
    page_metadata: PageMetadata
    sequences: list[Sequence]