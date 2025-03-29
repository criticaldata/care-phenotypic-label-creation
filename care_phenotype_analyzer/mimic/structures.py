"""
MIMIC data structures module.

This module defines the core data structures used for MIMIC-IV data processing.
These structures provide type safety and documentation for the data processing pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Union
from enum import Enum

class Gender(str, Enum):
    """Enumeration for patient gender."""
    MALE = "M"
    FEMALE = "F"
    OTHER = "Other"
    UNKNOWN = "Unknown"

class AdmissionType(str, Enum):
    """Enumeration for admission types."""
    ELECTIVE = "ELECTIVE"
    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    NEWBORN = "NEWBORN"
    UNKNOWN = "UNKNOWN"

class ICUUnit(str, Enum):
    """Enumeration for ICU units."""
    CCU = "CCU"  # Coronary Care Unit
    CSRU = "CSRU"  # Cardiac Surgery Recovery Unit
    MICU = "MICU"  # Medical ICU
    NICU = "NICU"  # Neonatal ICU
    NWARD = "NWARD"  # Neonatal Ward
    SICU = "SICU"  # Surgical ICU
    TSICU = "TSICU"  # Trauma/Surgical ICU

@dataclass
class Patient:
    """Represents a patient in the MIMIC database."""
    subject_id: int
    gender: Gender
    anchor_age: int
    anchor_year: int
    anchor_year_group: str
    dod: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate patient data after initialization."""
        if self.anchor_age < 0 or self.anchor_age > 120:
            raise ValueError(f"Invalid age: {self.anchor_age}")
        if self.anchor_year < 1900 or self.anchor_year > datetime.now().year:
            raise ValueError(f"Invalid year: {self.anchor_year}")

@dataclass
class Admission:
    """Represents a hospital admission in the MIMIC database."""
    subject_id: int
    hadm_id: int
    admittime: datetime
    dischtime: datetime
    deathtime: Optional[datetime] = None
    admission_type: AdmissionType = AdmissionType.UNKNOWN
    admission_location: str = ""
    discharge_location: str = ""
    insurance: str = ""
    language: str = ""
    marital_status: str = ""
    ethnicity: str = ""
    
    def __post_init__(self):
        """Validate admission data after initialization."""
        if self.dischtime < self.admittime:
            raise ValueError("Discharge time cannot be before admission time")
        if self.deathtime and self.deathtime < self.admittime:
            raise ValueError("Death time cannot be before admission time")

@dataclass
class ICUStay:
    """Represents an ICU stay in the MIMIC database."""
    subject_id: int
    hadm_id: int
    stay_id: int
    intime: datetime
    outtime: datetime
    first_careunit: ICUUnit
    last_careunit: ICUUnit
    los: float  # Length of stay in days
    
    def __post_init__(self):
        """Validate ICU stay data after initialization."""
        if self.outtime < self.intime:
            raise ValueError("Out time cannot be before in time")
        if self.los < 0:
            raise ValueError("Length of stay cannot be negative")

@dataclass
class LabEvent:
    """Represents a laboratory event in the MIMIC database."""
    subject_id: int
    hadm_id: int
    stay_id: Optional[int]
    charttime: datetime
    specimen_id: int
    itemid: int
    valuenum: Optional[float]
    valueuom: Optional[str]
    ref_range_lower: Optional[float]
    ref_range_upper: Optional[float]
    flag: Optional[str] = None
    
    def __post_init__(self):
        """Validate lab event data after initialization."""
        if self.valuenum is not None:
            if self.ref_range_lower and self.valuenum < self.ref_range_lower:
                self.flag = f"{self.flag}|LOW" if self.flag else "LOW"
            if self.ref_range_upper and self.valuenum > self.ref_range_upper:
                self.flag = f"{self.flag}|HIGH" if self.flag else "HIGH"

@dataclass
class ChartEvent:
    """Represents a chart event in the MIMIC database."""
    subject_id: int
    hadm_id: int
    stay_id: int
    charttime: datetime
    storetime: datetime
    itemid: int
    value: str
    valuenum: Optional[float]
    valueuom: Optional[str]
    warning: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Validate chart event data after initialization."""
        if self.storetime < self.charttime:
            raise ValueError("Store time cannot be before chart time")

@dataclass
class ClinicalScore:
    """Represents a clinical score calculation."""
    subject_id: int
    hadm_id: int
    stay_id: Optional[int]
    score_time: datetime
    score_type: str
    score_value: float
    components: Dict[str, float]
    
    def __post_init__(self):
        """Validate clinical score data after initialization."""
        if self.score_value < 0:
            raise ValueError("Score value cannot be negative")
        if sum(self.components.values()) != self.score_value:
            raise ValueError("Component sum does not match total score")

# Type aliases for collections
PatientCollection = List[Patient]
AdmissionCollection = List[Admission]
ICUStayCollection = List[ICUStay]
LabEventCollection = List[LabEvent]
ChartEventCollection = List[ChartEvent]
ClinicalScoreCollection = List[ClinicalScore]

# Dictionary types for quick lookups
PatientDict = Dict[int, Patient]
AdmissionDict = Dict[int, Admission]
ICUStayDict = Dict[int, ICUStay]
LabEventDict = Dict[int, List[LabEvent]]
ChartEventDict = Dict[int, List[ChartEvent]]
ClinicalScoreDict = Dict[str, List[ClinicalScore]] 