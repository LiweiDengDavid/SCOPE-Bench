import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const siteRoot = new URL("../", import.meta.url);

test("exports the SCOPE-Bench project page", async () => {
  const html = await readFile(new URL("dist/client/index.html", siteRoot), "utf8");
  assert.match(html, /SCOPE-Bench/);
  assert.match(html, /Rethinking the attention economy/i);
  assert.match(html, /Content Depth Score/);
  assert.match(html, /A-LCDS/);
  assert.match(html, /Overview of the scalable CDS annotation workflow/);
  assert.match(html, /individual short videos from the user perspective/);
  assert.match(html, /Overview of the content-depth-aware RS evaluation framework/);
  assert.match(html, /recommendation lists from the platform perspective/);
  assert.match(html, /One benchmark, four settings/);
  assert.doesNotMatch(html, /What depth does a video contain|What depth does a recommender deliver/);
  assert.doesNotMatch(html, /One benchmark, five settings/);
  assert.match(html, /More work is on the way/);
  assert.match(html, /Liwei Deng/);
  assert.match(html, /https:\/\/liweidengdavid\.github\.io\//);
  assert.match(html, /https:\/\/zhw\.li\//);
  assert.match(html, /~Yang_Wang134/);
  assert.match(html, /Australian Artificial Intelligence Institute/);
  assert.match(html, /13 representative RSs/);
  assert.match(html, /Coming soon/);
  assert.doesNotMatch(html, /14 representative paper-facing baselines|Cite this work/);
  assert.match(html, /v1\.0/);
  assert.doesNotMatch(html, /2\.0|3\.0|Coming next|On the roadmap/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
});

test("includes required public assets", async () => {
  await Promise.all([
    access(new URL("dist/client/.nojekyll", siteRoot)),
    access(new URL("public/favicon.svg", siteRoot)),
    access(new URL("public/figures/overview-b.png", siteRoot)),
    access(new URL("public/og.png", siteRoot)),
    access(new URL("public/branding/uts-logo-wide.png", siteRoot)),
  ]);

  const publishedCss = await readdir(
    new URL("dist/client/_next/static/css/", siteRoot),
  );
  assert.ok(publishedCss.some((file) => file.endsWith(".css")));
});
