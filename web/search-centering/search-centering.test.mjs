// Regression test for https://github.com/yukakust/GPU/issues/2
// "Not in center title": the project search placeholder sits below the centre
// of the search pill because the input inherits margin-top:8px from the generic
// `label>input:not([type=checkbox])` form rule.
//
// Run:        node --test web/search-centering/
// Live check: GPU_LIVE_URL=https://gpu.social/ node --test web/search-centering/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright';

const dir = path.dirname(fileURLToPath(import.meta.url));
const fixtureUrl = pathToFileURL(path.join(dir, 'fixture.html')).href;
const fixCss = path.join(dir, 'fix.css');

async function measure(page) {
  return page.evaluate(() => {
    const label = document.querySelector('label.search');
    const input = label.querySelector('input[type=search]');
    label.scrollIntoView({ block: 'center' });
    const l = label.getBoundingClientRect();
    const i = input.getBoundingClientRect();
    return {
      marginTop: getComputedStyle(input).marginTop,
      offset: (i.top + i.height / 2) - (l.top + l.height / 2),
      overflowBottom: i.bottom - l.bottom,
    };
  });
}

async function withPage(fn) {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    return await fn(page);
  } finally {
    await browser.close();
  }
}

test('original CSS reproduces the off-centre search input', { timeout: 30_000 }, () =>
  withPage(async (page) => {
    await page.goto(fixtureUrl);
    const m = await measure(page);
    assert.equal(m.marginTop, '8px');
    assert.ok(m.offset >= 3, `expected input to be pushed down, offset=${m.offset}`);
  }));

test('fix centres the search input inside the pill', { timeout: 30_000 }, () =>
  withPage(async (page) => {
    await page.goto(fixtureUrl);
    await page.addStyleTag({ path: fixCss });
    const m = await measure(page);
    assert.equal(m.marginTop, '0px');
    assert.ok(Math.abs(m.offset) <= 0.5, `input not centred, offset=${m.offset}`);
    assert.ok(m.overflowBottom <= 0, `input overflows the pill by ${m.overflowBottom}px`);
  }));

test('live site search input is centred', {
  timeout: 60_000,
  skip: !process.env.GPU_LIVE_URL && 'set GPU_LIVE_URL to check the deployed site',
}, () =>
  withPage(async (page) => {
    await page.goto(process.env.GPU_LIVE_URL, { waitUntil: 'domcontentloaded' });
    const m = await measure(page);
    assert.equal(m.marginTop, '0px');
    assert.ok(Math.abs(m.offset) <= 0.5, `input not centred, offset=${m.offset}`);
  }));
