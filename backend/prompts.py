"""
Compatibility wrapper for the active institution's answer prompt.

The IIT Jodhpur implementation currently lives at:

    backend.institutions.iitj.prompts

Existing imports such as:

    from backend.prompts import answer_prompt

continue to work unchanged.
"""

from backend.institutions.iitj.prompts import answer_prompt

__all__ = [
    "answer_prompt",
]
