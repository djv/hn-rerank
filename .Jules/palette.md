## 2024-05-22 - [Enhancing Icon-Only Button Accessibility]
**Learning:** Icon-only buttons (like links styled as buttons) are often invisible to screen readers if they rely solely on SVG content or `title` attributes. `title` attributes are unreliable for accessibility.
**Action:** Always add `aria-label` to icon-only buttons/links to provide a clear text alternative. Also, mark the inner SVG as `aria-hidden="true"` to prevent it from being announced as a separate, potentially confusing element. Adding `rel="noopener noreferrer"` to `target="_blank"` links is a security best practice that also slightly improves performance.

## 2024-05-22 - [Stale Tests in Legacy Code]
**Learning:** When improving legacy code (or code that has evolved without test updates), you might find tests asserting on features that no longer exist (e.g., raw comment rendering vs. TLDR).
**Action:** It is necessary to clean up or update these stale tests to reflect the current reality of the codebase before or while applying new changes. Don't be afraid to remove tests for features that are clearly gone.
