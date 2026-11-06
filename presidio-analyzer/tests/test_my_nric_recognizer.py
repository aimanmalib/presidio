"""Tests for the Malaysian NRIC (MyKad) recognizer."""

from pathlib import Path

import pytest
import yaml

import presidio_analyzer
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import MyNricRecognizer
from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

# All numbers below are synthetic.


@pytest.fixture(scope="module")
def recognizer():
    return MyNricRecognizer()


@pytest.fixture(scope="module")
def entities():
    return ["MY_NRIC"]


@pytest.mark.parametrize(
    "text, expected",
    [
        # Valid: dashed, spaced and compact forms keep their pattern score
        ("900101-14-5678", [(0, 14, 0.5)]),
        ("900101 14 5678", [(0, 14, 0.5)]),
        ("900101145678", [(0, 12, 0.3)]),
        ("No IC: 900101-14-5678.", [(7, 21, 0.5)]),
        ("Nombor kad pengenalan 851231 10 6543 sahaja", [(22, 36, 0.5)]),
        # Valid: place-of-birth codes from each JPN series
        ("750615-01-1235", [(0, 14, 0.5)]),  # Johor
        ("750615-16-1235", [(0, 14, 0.5)]),  # Putrajaya
        ("750615-45-1235", [(0, 14, 0.5)]),  # older series
        ("750615-60-1235", [(0, 14, 0.5)]),  # born abroad
        ("750615-99-1235", [(0, 14, 0.5)]),  # code 99
        # Valid: 29 February only exists in 2000, not in 1900
        ("000229-10-1234", [(0, 14, 0.5)]),
        # Two numbers in one text
        ("IC 900101-14-5678 dan 020304-10-1235", [(3, 17, 0.5), (22, 36, 0.5)]),
        # Invalid date
        ("901301-14-5678", []),  # month 13
        ("900132-14-5678", []),  # day 32
        ("900431-14-5678", []),  # 31 April
        ("010229-10-1234", []),  # 2001 and 1901 are not leap years
        ("000000-10-1234", []),
        # Unassigned place-of-birth codes
        ("900101-00-5678", []),
        ("900101-17-5678", []),
        ("900101-20-5678", []),
        ("900101-69-5678", []),
        ("900101-70-5678", []),
        ("900101-73-5678", []),
        ("900101-80-5678", []),
        ("900101-81-5678", []),
        ("900101-94-5678", []),
        ("900101-97-5678", []),
        # Part of a longer number
        ("1900101-14-5678", []),
        ("900101-14-56789", []),
        ("900101-14-5678-01", []),
        ("9001011456781", []),
        ("4111 900101 14 5678", []),
        # Lookalikes: SSM company numbers (12-digit format since 2019) start
        # with the registration year, so the "month" is 19 or more
        ("SSM 201901000005", []),
        ("No. pendaftaran 202403123456", []),
        # Lookalike: a 12-digit timestamp
        ("ts=202611061230", []),
        # Wrong grouping
        ("90010-114-5678", []),
        ("900101/14/5678", []),
    ],
)
def test_when_nric_in_text_then_all_nrics_found(text, expected, recognizer, entities):
    results = sorted(recognizer.analyze(text, entities), key=lambda r: r.start)
    assert [(r.start, r.end) for r in results] == [(s, e) for s, e, _ in expected]
    for result, (_, _, score) in zip(results, expected):
        assert result.entity_type == "MY_NRIC"
        assert result.score == pytest.approx(score)


@pytest.mark.parametrize(
    "number, invalid",
    [
        ("900101-14-5678", False),
        ("900101145678", False),
        ("000229 10 1234", False),
        ("010229-10-1234", True),
        ("900101-18-5678", True),
        ("90010114567", True),
    ],
)
def test_invalidate_result(recognizer, number, invalid):
    assert recognizer.invalidate_result(number) is invalid


def test_valid_pb_codes_match_jpn_list():
    """86 codes: 01-16, 21-68, 71-72, 74-79, 82-93, 98 and 99."""
    codes = MyNricRecognizer.VALID_PB_CODES
    assert len(codes) == 86
    unassigned = ["17", "18", "19", "20", "69", "70", "73", "80", "81"]
    unassigned += ["94", "95", "96", "97"]
    assert not codes.intersection(unassigned)
    assert all(len(code) == 2 and code.isdigit() for code in codes)


def test_context_has_no_bare_short_abbreviation(recognizer):
    # "ic" alone would match "public" or "clinic" in substring mode.
    assert "ic" not in recognizer.context
    assert {"mykad", "nric", "kad pengenalan", "no ic"} <= set(recognizer.context)


def _shipped_entry():
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "MyNricRecognizer"]
    assert len(entries) == 1, "MyNricRecognizer missing from YAML"
    return entries[0]


def test_shipped_yaml_entry_is_disabled_country_specific():
    entry = _shipped_entry()
    assert entry["country_code"] == "my"
    assert entry["enabled"] is False
    assert entry["supported_languages"] == ["en"]


@pytest.fixture(scope="module")
def analyzer_from_yaml(spacy_nlp_engine):
    """Analyzer built the way users enable the recognizer: through YAML."""
    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(_shipped_entry(), enabled=True)],
        }
    ).create_recognizer_registry()
    assert [type(r).__name__ for r in registry.recognizers] == ["MyNricRecognizer"]
    return AnalyzerEngine(
        registry=registry, nlp_engine=spacy_nlp_engine, supported_languages=["en"]
    )


@pytest.mark.parametrize(
    "text, expected_span, expected_score",
    [
        # No context: pattern score
        ("Customer 900101-14-5678 called today", "900101-14-5678", 0.5),
        ("Customer 900101145678 called today", "900101145678", 0.3),
        # Context before the number raises the score
        ("Her MyKad number is 900101-14-5678", "900101-14-5678", 0.85),
        ("NRIC: 900101145678", "900101145678", 0.65),
        ("Sila isi kad pengenalan 900101-14-5678", "900101-14-5678", 0.85),
        ("No IC 900101 14 5678", "900101 14 5678", 0.85),
        # "public" contains "ic" but is not context
        ("The public record 900101-14-5678", "900101-14-5678", 0.5),
        # Context is prefix-only by default: a word after the number does not count
        ("900101-14-5678 is my MyKad number", "900101-14-5678", 0.5),
    ],
)
def test_yaml_enabled_recognizer_detects_with_context(
    analyzer_from_yaml, text, expected_span, expected_score
):
    results = analyzer_from_yaml.analyze(text, language="en", entities=["MY_NRIC"])
    assert len(results) == 1
    result = results[0]
    assert text[result.start : result.end] == expected_span
    assert result.score == pytest.approx(expected_score)


def test_yaml_enabled_recognizer_ignores_ssm_number(analyzer_from_yaml):
    text = "Syarikat berdaftar SSM 201901000005"
    assert analyzer_from_yaml.analyze(text, language="en", entities=["MY_NRIC"]) == []


def test_country_filter_keeps_malaysian_recognizer():
    """supported_countries ["my"] keeps MyNric and drops other countries' recognizers."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    shipped = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    by_name = {r.get("name"): r for r in shipped}
    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "supported_countries": ["my"],
            "recognizers": [
                dict(by_name["MyNricRecognizer"], enabled=True),
                dict(by_name["SgFinRecognizer"], enabled=True),
            ],
        }
    ).create_recognizer_registry()
    assert [type(r).__name__ for r in registry.recognizers] == ["MyNricRecognizer"]
