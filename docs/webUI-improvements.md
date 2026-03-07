# Web UI Improvements

Potential improvements to the Wildlife Monitor web UI, grouped by effort.

## Quick Wins

- [x] **Date filter on gallery** — Add a date range picker to the gallery filter bar; currently only trigger type and animal class are filterable
- [x] **Infinite scroll** — Replace the manual "Load More" button in the gallery with intersection-observer auto-loading
- [x] **Use toast notifications** — `showToast()` is fully implemented in `app.js` but unused; replace `alert()` calls in gallery, highlights, and detection pages
- [x] **Active nav link** — Highlight the current page in the navbar based on `window.location.pathname`
- [x] **Animal emoji icons** — `getAnimalIcon()` is implemented in `app.js` but never displayed; show emoji in gallery cards and detection detail

## Medium Effort

- [ ] **Gallery sort options** — Currently fixed newest-first; add sort by confidence or animal type
- [ ] **Keyboard navigation** — Left/right arrow keys to navigate between detections on the detail page
- [ ] **Empty state illustrations** — Replace plain "No detections" text with a simple SVG or emoji-based empty state
- [ ] **Relative timestamps** — `formatRelativeTime()` exists in `app.js` but gallery cards show full timestamps; use "2 hours ago" style instead
- [ ] **Video autoplay on detection page** — Autoplay (muted) on the detection detail page to save a click

## Larger Features

- [ ] **Date range filter on statistics** — Replace the "last N days" dropdown with a proper date picker for more control
- [ ] **Hourly activity heatmap** — Time-of-day × day-of-week grid showing which hours have the most detections
- [ ] **Gallery calendar view** — Alternative view showing detection counts per day in a calendar layout
- [ ] **Comparison view** — Side-by-side video playback for two detections
- [ ] **PWA support** — Add a web app manifest and service worker so the UI can be installed on a phone's home screen when accessing the Pi on the local network
