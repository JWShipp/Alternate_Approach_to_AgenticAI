from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Minimal common aliases; extend via user mapping CSV.
DEFAULT_ALIASES: Dict[str, str] = {
    "RUSSIA": "RUS",
    "RUSSIAN FEDERATION": "RUS",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "IRAN": "IRN",
    "IRAN, ISLAMIC REPUBLIC OF": "IRN",
    "NORTH KOREA": "PRK",
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": "PRK",
    "SOUTH KOREA": "KOR",
    "KOREA, REPUBLIC OF": "KOR",
    "UNITED KINGDOM": "GBR",
    "GREAT BRITAIN": "GBR",
    "CHINA": "CHN",
    "PEOPLE'S REPUBLIC OF CHINA": "CHN",
}

@dataclass
class ISO3Mapper:
    alias_to_iso3: Dict[str, str]

    @classmethod
    def from_optional_csv(cls, mapping_csv: Optional[Path] = None) -> "ISO3Mapper":
        alias = dict(DEFAULT_ALIASES)
        if mapping_csv and Path(mapping_csv).exists():
            df = pd.read_csv(mapping_csv)
            # Expected columns: alias, iso3
            if "alias" in df.columns and "iso3" in df.columns:
                for _, r in df.iterrows():
                    a = str(r["alias"]).strip().upper()
                    i = str(r["iso3"]).strip().upper()
                    if len(i) == 3:
                        alias[a] = i
        return cls(alias_to_iso3=alias)

    def normalize(self, value: str) -> str:
        if value is None:
            return ""
        s = str(value).strip()
        if len(s) == 3 and s.isalpha():
            return s.upper()
        s_up = s.upper()
        return self.alias_to_iso3.get(s_up, s_up[:3] if len(s_up) >= 3 else s_up)
