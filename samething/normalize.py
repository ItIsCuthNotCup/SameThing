"""Record ingestion and normalization.

Original values are preserved; normalized fields are added alongside them.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------------------
# Column mapping
# ----------------------------------------------------------------------------

@dataclass
class ColumnMap:
    id: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None  # column name, or None -> use default_currency
    default_currency: Optional[str] = None

    def to_dict(self):
        return asdict(self)


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "with", "in", "on", "to", "by",
    "from", "at", "is", "new", "black", "white", "silver",  # colours kept separately
}
COLOR_WORDS = {"black", "white", "silver", "red", "blue", "gray", "grey", "green", "pink", "gold", "titanium"}

UNIT_PATTERNS = [
    # (regex, canonical suffix, multiplier to canonical unit)
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(tb|terabytes?)\b", re.I), "gb", 1000.0),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(gb|gigabytes?)\b", re.I), "gb", 1.0),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(mb|megabytes?)\b", re.I), "gb", 0.001),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(ghz)\b", re.I), "mhz", 1000.0),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(mhz)\b", re.I), "mhz", 1.0),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(inch|inches|in\.|\"|'')\b", re.I), "in", 1.0),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(mp|megapixels?)\b", re.I), "mp", 1.0),
    (re.compile(r"(?<![a-z0-9\-])(\d+(?:\.\d+)?)\s*-?\s*(watts?|w)\b", re.I), "w", 1.0),
]

PACK_RE = re.compile(r"\b(\d+)\s*[- ]?\s*(pack|pk|pcs|pieces|count|ct)\b|\bpack of (\d+)\b|\bset of (\d+)\b|\b(\d+)\s*x\b", re.I)
CONDITION_RE = re.compile(r"\b(refurbished|refurb|renewed|used|pre-owned|open box|open-box|remanufactured)\b", re.I)
BUNDLE_RE = re.compile(r"\b(bundle|kit|with case|w/ case|combo|includes|included|\+)\b", re.I)
ACCESSORY_RE = re.compile(
    r"\b(case|cover|cable|adapter|charger|battery|batteries|mount|bracket|stand|remote|"
    r"filter|bag|strap|lens cap|lens hood|cartridge|ink|toner|belt|stylus|earpad|ear pad|"
    r"skin|screen protector|holster|dock|cradle|replacement)\b", re.I)

# Model-number-like tokens: contain digits and letters, or long digit runs with hyphens/slashes
MODEL_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9/\-\.]*\d)(?=[A-Za-z0-9/\-\.]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9/\-\.]{2,}[A-Za-z0-9]\b")
DIGIT_MODEL_RE = re.compile(r"\b\d{5,}(?:[-/]\d+)?\b")
# number + unit suffix tokens are specifications, not identifiers (1080p, 320gb, 54mbps, 120hz)
SPEC_TOKEN_RE = re.compile(
    r"^\d+(?:\.\d+)?(?:p|i|k|x|gb|mb|tb|kb|hz|khz|mhz|ghz|mbps|gbps|kbps|w|mm|cm|m|in|ft|mp|rpm|bit|bits|ohm|ohms|dpi|ppm|fps|mah|wh|v|a|lb|lbs|oz|hr|hrs|ms|nm|db|cd|awg|ch|pc|pcs|pk|ct|d|g|kg|l|ml|mm|sec|min|yr|deg|btu|cu|rms|pin|port|ports|bay|watt|watts|inch|disc|discs|pack|piece|pieces|way|zone|zones|sheet|sheets|page|pages|cup|cups|qt|gal|ton|speed|cyl|step|line|lines|hd|dvd|cd|hour|hours|day|days|month|months|year|years|cup|disc|port|channel|channels|player|players|piece|button|buttons|key|keys|sheet|slice|slices|quart|qt|amp|amps|stage|stages|position|positions|setting|settings|tray|trays|door|doors|drawer|drawers|burner|burners|element|elements|rack|racks|shelf|shelves|degree|degrees|foot|feet|user|users|room|rooms|person|people|seat|seats)$", re.I)
SPEC_TOKEN_RE2 = re.compile(r"^(?:mpeg|mp|usb|ddr|ieee|802\.?11|h|x|wi-?fi|dvi|hdmi|dts|dolby|ac|dc|v|rs|sata|pci|pcie|dvd|cd|blu|ray|core|duo|i|gen|ver|version|type|class|cat)\d+[a-z]*$", re.I)
COMMON_NUMBERS = {"1080", "1920", "720", "1280", "1024", "768", "480", "2000", "2007", "2008", "2009", "2010", "100", "1000", "3000", "5000", "10000"}
PRICE_RE = re.compile(r"([$€£])?\s*([0-9][0-9,]*(?:\.\d{1,2})?)")


def clean_text(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text(s: str) -> str:
    """Lower-case, normalize punctuation/whitespace, canonicalize units.

    Model identifiers are kept intact (hyphens and slashes within tokens survive
    as a separate model-token extraction; here we only collapse punctuation).
    """
    s = clean_text(s).lower()
    for rx, unit, mult in UNIT_PATTERNS:
        def _rep(m, unit=unit, mult=mult):
            v = float(m.group(1)) * mult
            v = int(v) if abs(v - round(v)) < 1e-9 else round(v, 3)
            return f" {v}{unit} "
        s = rx.sub(_rep, s)
    s = re.sub(r"[^a-z0-9/\-\.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str) -> list[str]:
    s = re.sub(r"[/\-\.]", " ", s)
    return [t for t in s.split() if t and t not in STOPWORDS]


def canon_model(tok: str) -> str:
    """Canonical model key: strip separators, lower. 'PS-LX350H' -> 'pslx350h'."""
    return re.sub(r"[\-/\. ]", "", tok.lower())


COMPAT_SPLIT_RE = re.compile(r"\b(for|fits|compatible with|works with|designed for|to fit)\b", re.I)


def strip_compatibility_clause(title: str) -> str:
    """'Add-On Handset For The KXTG6700B Phone' -> 'Add-On Handset'. Identifiers after 'for' describe another product."""
    m = COMPAT_SPLIT_RE.search(title)
    return title[: m.start()] if m and m.start() > 0 else title


def extract_model_tokens(*texts: str) -> list[str]:
    out = []
    for t in texts:
        t = clean_text(t)
        for m in MODEL_TOKEN_RE.findall(t):
            flat = re.sub(r"[\-/\. ]", "", m)
            if SPEC_TOKEN_RE.match(m) or SPEC_TOKEN_RE2.match(m) or SPEC_TOKEN_RE.match(flat) or SPEC_TOKEN_RE2.match(flat):
                continue
            c = canon_model(m)
            if len(c) >= 4 and any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c) and c not in out:
                out.append(c)
        for m in DIGIT_MODEL_RE.findall(t):
            c = canon_model(m)
            if c not in out and c not in COMMON_NUMBERS:
                out.append(c)
    return out


def prune_common_model_tokens(record_lists: list[list["Record"]], max_df: int = 8) -> dict[str, int]:
    """Drop 'model tokens' shared by many records in either catalog; identifiers are rare by nature.

    Uses only the catalogs themselves (no labels). Returns the pruned tokens with their frequencies.
    """
    pruned = {}
    for recs in record_lists:
        df = Counter(t for r in recs for t in set(r.model_tokens))
        common = {t for t, n in df.items() if n > max_df}
        pruned.update({t: df[t] for t in common})
    for recs in record_lists:
        for r in recs:
            r.model_tokens = [t for t in r.model_tokens if t not in pruned]
    return pruned


def extract_pack_qty(text: str) -> Optional[int]:
    m = PACK_RE.search(text)
    if not m:
        return None
    for g in m.groups():
        if g:
            try:
                q = int(g)
                return q if 1 < q < 1000 else None
            except ValueError:
                pass
    return None


def extract_condition(text: str) -> str:
    m = CONDITION_RE.search(text)
    if not m:
        return "unknown"
    w = m.group(1).lower()
    if w.startswith("refurb") or w in ("renewed", "remanufactured"):
        return "refurbished"
    if w in ("used", "pre-owned"):
        return "used"
    return "open_box"


def extract_capacities(text_norm: str) -> dict[str, set]:
    caps: dict[str, set] = {}
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)(gb|mhz|in|mp|w)\b", text_norm):
        caps.setdefault(m.group(2), set()).add(float(m.group(1)))
    return caps


def parse_price(v) -> tuple[Optional[float], Optional[str]]:
    s = clean_text(v)
    if not s:
        return None, None
    m = PRICE_RE.search(s)
    if not m:
        return None, None
    cur = {"$": "USD", "€": "EUR", "£": "GBP"}.get(m.group(1) or "", None)
    try:
        return float(m.group(2).replace(",", "")), cur
    except ValueError:
        return None, None


def brand_from_title(title: str) -> str:
    toks = clean_text(title).split()
    return toks[0].lower() if toks else ""


@dataclass
class Record:
    source: str
    rid: str
    title: str
    description: str
    brand: str
    model: str
    price_raw: str
    price: Optional[float]
    currency: Optional[str]
    # normalized
    title_norm: str = ""
    desc_norm: str = ""
    brand_norm: str = ""
    model_tokens: list = field(default_factory=list)
    desc_model_tokens: list = field(default_factory=list)
    compat_model_tokens: list = field(default_factory=list)
    explicit_model: str = ""
    pack_qty: Optional[int] = None
    condition: str = "unknown"
    capacities: dict = field(default_factory=dict)
    accessory_hint: bool = False
    bundle_hint: bool = False

    def visible_fields(self) -> dict:
        """The fields every matcher (rules, fuzzy, Jev) is allowed to see."""
        d = {"title": self.title}
        if self.description:
            d["description"] = self.description[:600]
        if self.brand:
            d["brand"] = self.brand
        if self.model:
            d["model"] = self.model
        if self.price_raw:
            d["price"] = self.price_raw
        return d


def build_records(df: pd.DataFrame, cmap: ColumnMap, source: str) -> list[Record]:
    recs = []
    for _, row in df.iterrows():
        title = clean_text(row[cmap.title])
        desc = clean_text(row[cmap.description]) if cmap.description else ""
        brand = clean_text(row[cmap.brand]) if cmap.brand else ""
        model = clean_text(row[cmap.model]) if cmap.model else ""
        price_raw = clean_text(row[cmap.price]) if cmap.price else ""
        price, cur = parse_price(price_raw)
        if cmap.currency and cmap.currency in row:
            cur = clean_text(row[cmap.currency]) or cur
        cur = cur or cmap.default_currency
        r = Record(source=source, rid=str(row[cmap.id]), title=title, description=desc,
                   brand=brand, model=model, price_raw=price_raw, price=price, currency=cur)
        r.title_norm = normalize_text(title)
        r.desc_norm = normalize_text(desc)
        r.brand_norm = normalize_text(brand) if brand else brand_from_title(title)
        r.explicit_model = canon_model(model) if model else ""
        core_title = strip_compatibility_clause(title)
        r.model_tokens = extract_model_tokens(model, core_title)
        r.compat_model_tokens = [t for t in extract_model_tokens(title) if t not in r.model_tokens]
        r.desc_model_tokens = [t for t in extract_model_tokens(desc) if t not in r.model_tokens]
        full = f"{title} {desc}"
        r.pack_qty = extract_pack_qty(full)
        r.condition = extract_condition(full)
        r.capacities = {k: sorted(v) for k, v in extract_capacities(r.title_norm).items()}
        r.accessory_hint = bool(ACCESSORY_RE.search(title))
        r.bundle_hint = bool(BUNDLE_RE.search(title))
        recs.append(r)
    return recs


def records_to_frame(recs: list[Record]) -> pd.DataFrame:
    rows = []
    for r in recs:
        d = asdict(r)
        d["model_tokens"] = " ".join(r.model_tokens)
        d["desc_model_tokens"] = " ".join(r.desc_model_tokens)
        d["compat_model_tokens"] = " ".join(r.compat_model_tokens)
        d["capacities"] = str(r.capacities)
        rows.append(d)
    return pd.DataFrame(rows)
