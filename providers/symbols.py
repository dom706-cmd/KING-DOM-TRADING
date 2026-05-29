from __future__ import annotations

def to_provider_symbol(sym: str) -> str | None:
    """Normalize a symbol to Alpaca's conventions.

    Returns:
        - normalized symbol string, or
        - None if the symbol should be skipped (no safe mapping)

    Rules:
        - Convert preferred/class-share separators to Alpaca dot notation.
        - Convert Provider dash notation to Alpaca dot notation (e.g., BRK-B -> BRK.B).
        - Allow only A-Z, 0-9, and '.' after normalization.
    """
    s = str(sym or "").strip().upper()
    if not s:
        return None

    s = s.replace("$", ".")
    s = s.replace("-", ".")

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
    if any(ch not in allowed for ch in s):
        return None

    return s if s else None


def to_massive_symbol(sym: str) -> str | None:
    """Normalize a symbol to Massive's conventions.

    Massive (formerly Polygon) generally uses dot notation for class shares (e.g., BRK.B).
    We also accept Provider-style dash class shares and convert them back to dots.

    Rules:
        - Skip preferred-share symbols containing '$' (no universal mapping)
        - Convert class-share dash notation to dot notation (BRK-B -> BRK.B)
        - Allow only A-Z, 0-9, '.', and '-' after normalization
    """
    s = sym.strip().upper()
    if "$" in s:
        return None

    # Convert Provider-style class-share dashes back to dot form
    # but only for the common single dash class-share pattern.
    if "-" in s and "." not in s:
        # e.g. BRK-B => BRK.B, BF-B => BF.B
        parts = s.split("-")
        if len(parts) == 2 and parts[1] in {"A","B","C","D"} and len(parts[0]) > 0:
            s = parts[0] + "." + parts[1]

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    if any(ch not in allowed for ch in s):
        return None
    return s if s else None
