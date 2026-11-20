"""MaiMuriDX headless extraction; importing this package performs no detection."""

from .api import DetectionError, detect_file, detect_simai, looks_like_maidata, parse_maidata

__all__ = ["DetectionError", "detect_file", "detect_simai", "looks_like_maidata", "parse_maidata"]
