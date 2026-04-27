"""Jurisdiction-specific regulatory modules for the credit workflow.

Each market exposes a class that the pipeline consults at runtime to
obtain the disclosure block, protected-attribute keyword list, fairness
threshold, and other region-specific knobs.

Currently implemented (10 markets):

    Americas     : :class:`US`, :class:`Canada`, :class:`Brazil`
    EMEA         : :class:`UK`, :class:`EU`, :class:`UAE`
    Asia-Pacific : :class:`India`, :class:`Singapore`, :class:`Japan`,
                   :class:`Australia`

Each module ships a paired template under ``jurisdictions/templates/``
that contains the appended legal disclosure text — plain text so
counsel can review and edit without touching Python.
"""
from .australia import AUSTRALIA_PROTECTED_KEYWORDS, Australia
from .base import ExplainabilityLevel, JurisdictionBase, load_template
from .brazil import BRAZIL_PROTECTED_KEYWORDS, Brazil
from .canada import CANADA_PROTECTED_KEYWORDS, Canada
from .eu import EU, EU_PROTECTED_KEYWORDS
from .india import INDIA_PROTECTED_KEYWORDS, India
from .japan import JAPAN_PROTECTED_KEYWORDS, Japan
from .singapore import SINGAPORE_PROTECTED_KEYWORDS, Singapore
from .uae import UAE, UAE_PROTECTED_KEYWORDS
from .uk import UK, UK_PROTECTED_KEYWORDS
from .us import US, US_PROTECTED_KEYWORDS

#: Registry of every implemented jurisdiction class, keyed by ISO code.
#: Useful for config-driven instantiation: ``ALL_JURISDICTIONS[cfg.code]()``.
ALL_JURISDICTIONS = {
    "US": US,
    "UK": UK,
    "EU": EU,
    "IN": India,
    "CA": Canada,
    "AU": Australia,
    "SG": Singapore,
    "JP": Japan,
    "AE": UAE,
    "BR": Brazil,
}

__all__ = [
    "ExplainabilityLevel",
    "JurisdictionBase",
    "load_template",
    "ALL_JURISDICTIONS",
    # Jurisdiction classes
    "US",
    "UK",
    "EU",
    "India",
    "Canada",
    "Australia",
    "Singapore",
    "Japan",
    "UAE",
    "Brazil",
    # Keyword lists (importable for downstream tooling / config validation)
    "US_PROTECTED_KEYWORDS",
    "UK_PROTECTED_KEYWORDS",
    "EU_PROTECTED_KEYWORDS",
    "INDIA_PROTECTED_KEYWORDS",
    "CANADA_PROTECTED_KEYWORDS",
    "AUSTRALIA_PROTECTED_KEYWORDS",
    "SINGAPORE_PROTECTED_KEYWORDS",
    "JAPAN_PROTECTED_KEYWORDS",
    "UAE_PROTECTED_KEYWORDS",
    "BRAZIL_PROTECTED_KEYWORDS",
]
