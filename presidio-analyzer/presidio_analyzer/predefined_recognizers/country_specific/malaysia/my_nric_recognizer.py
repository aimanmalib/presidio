from datetime import date
from typing import FrozenSet, List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class MyNricRecognizer(PatternRecognizer):
    """
    Recognize Malaysian NRIC (MyKad) numbers.

    The MyKad number has 12 digits, written ``YYMMDD-PB-###G``:

    - ``YYMMDD`` is the holder's date of birth.
    - ``PB`` is the place-of-birth code assigned by the National Registration
      Department (JPN): a state code (01-16), a state code from the older
      series (21-59) or a country or region code for people born abroad.
    - ``###G`` is a serial number; the last digit is odd for men and even
      for women.

    There is no check digit, so validation is structural: ``YYMMDD`` must be
    a real calendar date in the 1900s or the 2000s (``000229`` is accepted
    only as 2000-02-29) and ``PB`` must be one of the codes published by JPN.
    A 12-digit run that passes these checks is still often an account,
    membership or policy number, so a match keeps its pattern score (0.5
    dashed or spaced, 0.3 compact) and relies on context words to rise.
    The checks reject about 97% of random 12-digit runs, and every SSM
    company number in the 12-digit format introduced in 2019 (it starts
    with the registration year, so the "month" would be 19 or more).

    Reference: https://www.jpn.gov.my/my/kod-negeri
               https://www.jpn.gov.my/my/kod-negara
               https://en.wikipedia.org/wiki/Malaysian_identity_card

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param name: Name of the recognizer
    """

    COUNTRY_CODE = "my"

    # Place-of-birth codes published by JPN (kod-negeri and kod-negara).
    # 17-20, 69, 70, 73, 80, 81, 94-97 are not assigned.
    VALID_PB_CODES: FrozenSet[str] = frozenset(
        [f"{n:02d}" for n in range(1, 17)]  # states and federal territories
        + [f"{n:02d}" for n in range(21, 69)]  # older state series, then abroad
        + ["71", "72", "74", "75", "76", "77", "78", "79"]
        + [f"{n:02d}" for n in range(82, 94)]
        + ["98", "99"]
    )

    PATTERNS = [
        Pattern(
            "MY_NRIC dashed (medium)",
            r"(?<![\d-])\d{6}-\d{2}-\d{4}(?![\d-])",
            0.5,
        ),
        Pattern(
            "MY_NRIC spaced (medium)",
            # Not preceded by a group of digits, so a longer spaced number
            # such as a card or account number does not match in its middle.
            r"(?<!\d)(?<!(?:^|\W)\d+ )\d{6} \d{2} \d{4}(?! ?\d)",
            0.5,
        ),
        Pattern(
            "MY_NRIC compact (weak)",
            r"(?<!\d)\d{12}(?!\d)",
            0.3,
        ),
    ]

    # Multi-word phrases rather than the bare abbreviation "ic", which in the
    # default substring mode would also match "public" or "clinic".
    CONTEXT = [
        "mykad",
        "nric",
        "kad pengenalan",
        "identity card",
        "ic no",
        "no ic",
        "nombor ic",
        "no kp",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "MY_NRIC",
        name: Optional[str] = None,
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        """
        Reject numbers with an impossible birth date or place-of-birth code.

        A structurally valid number keeps its pattern score instead of being
        raised to the maximum: without a check digit, passing these checks
        does not make a 12-digit number a MyKad number.

        :param pattern_text: the text detected by the regex engine
        :return: True if the number cannot be a MyKad number
        """
        digits = "".join(c for c in pattern_text if c.isdigit())
        if len(digits) != 12:
            return True
        if digits[6:8] not in self.VALID_PB_CODES:
            return True
        return not self._is_valid_birth_date(digits[:6])

    @staticmethod
    def _is_valid_birth_date(yymmdd: str) -> bool:
        yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        for century in (1900, 2000):
            try:
                date(century + yy, mm, dd)
                return True
            except ValueError:
                continue
        return False
