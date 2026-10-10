# Search placeholder centering (issue #2)

Fix for [yukakust/GPU#2](https://github.com/yukakust/GPU/issues/2): on
[gpu.social](https://gpu.social/#projects) the placeholder of the project search
box ("Найти проект" / "Find a project") is not vertically centred.

| Before | After |
| --- | --- |
| ![before](../../docs/screenshots/issue-2-search-before.png) | ![after](../../docs/screenshots/issue-2-search-after.png) |

## Root cause

The search box is a `<label class="search">` wrapping `<input type="search">`.
`/workshop/style.css` has a generic form rule that also matches it:

```css
label>input:not([type=checkbox]),label>textarea{display:block;width:100%;margin-top:8px;…}
```

`.search>input[type=search]` overrides padding, border and background, but never
resets `margin`, so the input keeps `margin-top:8px`. With `height:100%` inside a
46px flex pill, the input is pushed down: its centre sits 4px below the pill's
centre and its bottom edge overflows the pill's border (measured on the live site
with Chromium: input centre 1045px vs pill centre 1041px).

## Fix

Add `margin:0` to the search input rule in `/workshop/style.css`
(see [`style.css.patch`](style.css.patch)):

```diff
-.search>input[type=search]{flex:1;min-width:0;height:100%;padding:0;border:0;background:none;outline:0;font-size:16px;color:var(--text)}
+.search>input[type=search]{flex:1;min-width:0;height:100%;margin:0;padding:0;border:0;background:none;outline:0;font-size:16px;color:var(--text)}
```

The selector is scoped to the search field, so other form inputs (e.g. the
"Добавить проект" form) keep their `margin-top:8px`.

## Tests

```sh
cd web/search-centering
npm install && npx playwright install chromium
npm test                                      # fixture: bug reproduced + fix verified
GPU_LIVE_URL=https://gpu.social/ npm test     # also check the deployed site
```

- `fixture.html` — minimal copy of the live markup and the relevant CSS rules.
- `fix.css` — the fix, applied on top of the fixture by the test.
- `search-centering.test.mjs` — measures the input's centre against the pill's
  centre; the live check fails until the patched `style.css` is deployed.
