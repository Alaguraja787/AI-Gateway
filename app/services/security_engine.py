"""
Security Engine
Pre-inference security checks: prompt injection, jailbreaks, PII leakage, unsafe instructions.
Runs BEFORE any model is called — block or flag before spending compute.
"""

import re
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"

@dataclass
class SecurityResult:
    threat_level: ThreatLevel
    blocked: bool
    threats_detected: list[str]
    pii_detected: list[str]
    sanitized_query: str
    security_score: float  # 0.0 = dangerous, 1.0 = safe

INJECTION_PATTERNS = [
    (r"ignore (all |previous |your )?(instructions|rules|guidelines)", "prompt_injection"),
    (r"you are now (a|an) (different|new|evil|unrestricted)", "role_hijacking"),
    (r"forget (everything|all|your|previous)", "context_poisoning"),
    (r"pretend (you are|to be|that)", "persona_hijacking"),
    (r"(act as|roleplay as|simulate being) (an AI|a system|GPT|DAN)", "jailbreak_persona"),
    (r"DAN|do anything now|jailbreak|bypass (safety|filter|restriction)", "jailbreak_attempt"),
    (r"<(script|iframe|img|svg)[^>]*>", "html_injection"),
    (r"\{\{.*\}\}|<%.*%>|\$\{.*\}", "template_injection"),
    (r"(system prompt|internal instructions|training data):?\s*(reveal|show|print|expose)", "prompt_leakage"),
]

UNSAFE_CONTENT_PATTERNS = [
    (r"(make|create|build|synthesize).*(bomb|explosive|weapon|malware|virus|toxin)", "harmful_content"),
    (r"how to (hack|exploit|crack|phish|ddos)", "cyberattack_guidance"),
    (r"(child|minor).*(explicit|sexual|nsfw)", "csam"),
    (r"(kill|murder|harm).*(person|people|individual)", "violence_incitement"),
]

PII_PATTERNS = [
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "ssn"),
    (r"\b(?:\d{4}[-\s]?){4}\b", "credit_card"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_address"),
    (r"\b\d{10,12}\b", "phone_number"),
    (r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", "password_in_query"),
]

def run_security_check(query: str) -> SecurityResult:
    threats = []
    pii_found = []
    threat_level = ThreatLevel.SAFE
    sanitized = query

    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            threats.append(label)

    for pattern, label in UNSAFE_CONTENT_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            threats.append(label)

    for pattern, label in PII_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            pii_found.append(label)
            sanitized = re.sub(pattern, f"[{label.upper()}_REDACTED]", sanitized, flags=re.IGNORECASE)

    blocked = any(t in threats for t in [
        "harmful_content", "cyberattack_guidance", "csam", "violence_incitement",
        "jailbreak_attempt", "jailbreak_persona"
    ])

    if blocked or any(t in threats for t in ["prompt_injection", "role_hijacking", "context_poisoning"]):
        threat_level = ThreatLevel.BLOCKED if blocked else ThreatLevel.HIGH
    elif threats:
        threat_level = ThreatLevel.MEDIUM
    elif pii_found:
        threat_level = ThreatLevel.LOW
    else:
        threat_level = ThreatLevel.SAFE

    threat_penalty = len(threats) * 0.15
    pii_penalty = len(pii_found) * 0.05
    security_score = max(0.0, round(1.0 - threat_penalty - pii_penalty, 2))

    return SecurityResult(
        threat_level=threat_level,
        blocked=blocked,
        threats_detected=threats,
        pii_detected=pii_found,
        sanitized_query=sanitized,
        security_score=security_score,
    )
