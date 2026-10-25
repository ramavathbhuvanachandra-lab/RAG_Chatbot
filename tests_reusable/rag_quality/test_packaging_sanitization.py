"""Regression tests for markdown link and navigation sanitization in packaging.py."""
import pytest
from ai_platform.core.evidence.packaging import sanitize_evidence_text


class TestPackagingSanitization:
    def test_ordinary_markdown_link(self):
        text = "Please see [the guidelines](https://example.com/guide) for details."
        expected = "Please see the guidelines for details."
        assert sanitize_evidence_text(text) == expected

    def test_nested_brackets_in_markdown_link(self):
        text = "Visit [ [Official Portal] ](https://iitj.ac.in/portal) for application."
        expected = "Visit Official Portal for application."
        assert sanitize_evidence_text(text) == expected

    def test_encoded_url_anchor(self):
        text = "[Section 22.5](https://www.iitj.ac.in/office-of-academics#top22.5%20Academic) specifies credits."
        expected = "Section 22.5 specifies credits."
        assert sanitize_evidence_text(text) == expected

    def test_irrelevant_navigation_wrapper_removal(self):
        text = "[ [Back to index] ](https://www.iitj.ac.in/office-of-academics/en/academic-regulations#top22.5%20Academic) Minimum credit requirements for M.Sc."
        expected = "Minimum credit requirements for M.Sc."
        assert sanitize_evidence_text(text) == expected

    def test_back_to_top_navigation_wrapper_removal(self):
        text = "Policy overview. [Back to top](https://iitj.ac.in/top) Next section."
        expected = "Policy overview. Next section."
        assert sanitize_evidence_text(text) == expected

    def test_plain_text_untouched(self):
        text = "60% or 6.0/10 for GEN candidates (55% for SC/ST)"
        assert sanitize_evidence_text(text) == text

    def test_bare_url_stripped(self):
        text = "Refer to the portal https://iitj.ac.in/admissions for updates."
        expected = "Refer to the portal for updates."
        assert sanitize_evidence_text(text) == expected
