## 2026-01-11 - Accessibility & Security Polish
**Learning:** External links (`target="_blank"`) in generated HTML templates were missing `rel="noopener noreferrer"`, posing a security risk. Icon-only buttons lacked `aria-label` and had low contrast/small touch targets.
**Action:** When working with static HTML generators, always audit templates for:
1. `rel="noopener noreferrer"` on external links.
2. `aria-label` on icon buttons.
3. Minimum contrast ratios (prefer `text-stone-600+` on `bg-stone-100`).
4. Minimum touch targets (use `p-2` or specific sizing).
