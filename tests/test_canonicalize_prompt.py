"""Tests for the OOD-prompt auto-canonicalizer.

The model is trained ONLY on prompts of the form
    "Locate all the instances that matches the following description: X."
Any other phrasing triggers MTP mode-collapse (the <ref>$$$$$ loop).
canonicalize_prompt() rewrites common natural-language phrasings; these tests
lock down which inputs it handles and which it leaves alone.
"""
import pytest
from lrai_locate_anything.orchestrator import canonicalize_prompt


class TestAlreadyCanonical:
    def test_passes_through(self):
        p = "Locate all the instances that matches the following description: cat."
        out, was = canonicalize_prompt(p)
        assert out == p
        assert was is False

    def test_case_insensitive_passthrough(self):
        p = "Locate ALL the instances that MATCHES the following DESCRIPTION: dog."
        out, was = canonicalize_prompt(p)
        assert was is False  # already canonical-shaped


class TestDetectVerb:
    def test_detect_all_X(self):
        out, was = canonicalize_prompt("Detect all cats. Return bounding boxes.")
        assert was is True
        assert out.startswith("Locate all the instances that matches the following description:")
        assert "cat" in out  # plural or singular both fine
        assert "Return bounding boxes" not in out

    def test_detect_X(self):
        out, was = canonicalize_prompt("Detect cars.")
        assert was is True
        assert "car" in out and out.startswith("Locate all the instances")

    def test_detect_every_X(self):
        out, was = canonicalize_prompt("Detect every dog.")
        assert was is True
        assert "dog" in out

    def test_detect_multi_class_and(self):
        out, was = canonicalize_prompt("Detect all cats and dogs.")
        assert was is True
        assert "</c>" in out
        assert "cat" in out and "dog" in out


class TestFindVerb:
    def test_find_all(self):
        out, was = canonicalize_prompt("Find all people.")
        assert was is True
        assert "people" in out

    def test_find_every(self):
        out, was = canonicalize_prompt("Find every face you can see.")
        assert was is True
        # "you can see" should be stripped or kept after rewriting — the important
        # thing is the canonical prefix is there
        assert out.startswith("Locate all the instances that matches the following description:")


class TestLocateButNotCanonical:
    def test_locate_X(self):
        """A "Locate X." short form isn't canonical; should be rewritten."""
        out, was = canonicalize_prompt("Locate the chair nearest the window.")
        assert was is True
        assert out.startswith("Locate all the instances that matches the following description:")


class TestWhereVerb:
    def test_where_are(self):
        out, was = canonicalize_prompt("Where are the red cars?")
        assert was is True
        assert "red car" in out


class TestUnrecognized:
    def test_question_passes_through(self):
        """A descriptive prompt with no detection verb shouldn't get rewritten."""
        out, was = canonicalize_prompt("What animal is on the candy?")
        assert was is False
        assert out == "What animal is on the candy?"

    def test_empty_string(self):
        out, was = canonicalize_prompt("")
        assert was is False


class TestBoilerplateStripping:
    def test_return_bounding_boxes_stripped(self):
        out, _ = canonicalize_prompt("Detect cats. Return bounding boxes.")
        assert "Return bounding boxes" not in out
        assert "bounding box" not in out.lower()

    def test_provide_coordinates_stripped(self):
        out, _ = canonicalize_prompt("Detect cars. Please provide coordinates.")
        assert "provide" not in out.lower()
        assert "coordinates" not in out.lower()


class TestMultiClassSeparator:
    def test_comma_join_converts_to_separator(self):
        out, was = canonicalize_prompt("Detect people, cars and bikes.")
        assert was is True
        # The </c> separator should appear between categories
        assert "</c>" in out

    def test_pure_and_join(self):
        out, was = canonicalize_prompt("Detect cats and dogs.")
        assert was is True
        # Should contain both classes joined by </c>
        assert "cat" in out
        assert "dog" in out
        assert "</c>" in out
