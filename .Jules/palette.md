## 2024-05-23 - Accessibility Improvements in HTML Templates
**Learning:** Hardcoded HTML templates in Python files often miss standard accessibility features like ARIA labels and proper contrast ratios because they lack the linting support found in dedicated frontend frameworks.
**Action:** When working with Python-generated HTML, explicitly check for:
1. ARIA labels on icon-only links.
2. `rel="noopener noreferrer"` on all `target="_blank"` links.
3. Color contrast ratios for metadata text (stone-400 is often too light on white).
