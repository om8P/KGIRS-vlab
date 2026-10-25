"""Light/dark colour palettes and the entity-type colour, shape and highlight tables."""

from __future__ import annotations

import streamlit as st


class Theme:
    """Provides high-contrast, WCAG-compliant styling tokens for Light and Dark modes."""
    def __init__(self, dark: bool = False):
        self.dark = dark
        if dark:
            self.ink = "#F8FAFC"             # Crisp off-white text
            self.muted = "#94A3B8"           # Medium slate for subtitles / cues
            self.edge = "#94A3B8"            # Clean slate line
            self.grid = "#334155"            # Subtle grid line
            self.halo = "rgba(15, 23, 42, 0.95)"  # Dark slate boundary for markers
            self.paper_bg = "rgba(0,0,0,0)"
            self.plot_bg = "rgba(0,0,0,0)"
            self.card_bg = "rgba(30, 41, 59, 0.65)"
            self.card_border = "rgba(148, 163, 184, 0.3)"
            self.card_accent = "#38BDF8"
            self.popup_bg = "#1E293B"        # Solid (not translucent) hover pop-up background
            self.badge_bg = "rgba(56, 189, 248, 0.18)"
            self.badge_text = "#38BDF8"
            self.node_colors = {
                "PERSON": "#38BDF8",     # Bright sky blue
                "ORG": "#FB923C",        # Bright coral orange
                "LOCATION": "#34D399",   # Vibrant emerald
                "DATE": "#C084FC",       # Vibrant violet
                "TERM": "#F87171",       # Salmon rose
                "MISC": "#A3A3A3",       # Neutral silver
            }
        else:
            self.ink = "#0F172A"             # Deep slate / navy text
            self.muted = "#475569"           # Slate for subtitles / cues
            self.edge = "#64748B"            # Rich slate edge
            self.grid = "#E2E8F0"            # Light grey grid line
            self.halo = "rgba(255, 255, 255, 0.95)" # White crisp boundary
            self.paper_bg = "rgba(0,0,0,0)"
            self.plot_bg = "rgba(0,0,0,0)"
            self.card_bg = "rgba(248, 250, 252, 0.85)"
            self.card_border = "rgba(100, 116, 139, 0.25)"
            self.card_accent = "#2563EB"
            self.popup_bg = "#FFFFFF"        # Solid (not translucent) hover pop-up background
            self.badge_bg = "rgba(37, 99, 235, 0.12)"
            self.badge_text = "#1D4ED8"
            self.node_colors = {
                "PERSON": "#2563EB",     # Classic royal blue
                "ORG": "#D97706",        # Amber orange
                "LOCATION": "#059669",   # Deep emerald
                "DATE": "#7C3AED",       # Deep violet
                "TERM": "#DC2626",       # Deep crimson
                "MISC": "#4B5563",       # Charcoal
            }

    def for_type(self, entity_type: str) -> str:
        return self.node_colors.get(entity_type, self.node_colors["MISC"])


def get_current_theme() -> Theme:
    """Detects the theme the viewer picked in Streamlit's own top-right "..." menu
    (System / Light / Dark), so charts and hover cards match the native page."""
    is_dark = False
    try:
        ctx_theme = getattr(st.context, "theme", None)
        if ctx_theme is not None and getattr(ctx_theme, "type", None):
            is_dark = (ctx_theme.type == "dark")
    except Exception:
        is_dark = False
    return Theme(dark=is_dark)


# Fixed marker shapes per entity type for double encoding (color + shape)
TYPE_SYMBOL = {
    "PERSON": "circle",
    "ORG": "square",
    "LOCATION": "diamond",
    "DATE": "triangle-up",
    "TERM": "hexagon",
    "MISC": "cross",
}


TYPE_SLOT = {
    "PERSON": 0,
    "ORG": 1,
    "LOCATION": 2,
    "DATE": 3,
    "TERM": 4,
    "MISC": 5,
}


# Streamlit Markdown text-highlight colors for annotated text view
TYPE_HIGHLIGHT = {
    "PERSON": "blue",
    "ORG": "orange",
    "LOCATION": "green",
    "DATE": "violet",
    "TERM": "red",
    "MISC": "gray",
}


ENTITY_TYPE_ORDER = ["PERSON", "ORG", "LOCATION", "DATE", "TERM", "MISC"]
