"use client";

import { useState } from "react";

const REPOSITORY = "https://github.com/LiweiDengDavid/SCOPE-Bench";

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <main>
      <header className="site-header">
        <div className="container header-inner">
          <a className="brand" href="#top" onClick={() => setMenuOpen(false)}>
            <span className="brand-mark" aria-hidden="true">◫</span>
            SCOPE-Bench <span className="version-badge">v1.0</span>
          </a>

          <button
            className="menu-button"
            type="button"
            aria-label="Toggle navigation"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <span />
            <span />
          </button>

          <nav className={menuOpen ? "open" : ""} aria-label="Primary navigation">
            <div className="top-nav">
              <a href="#use" onClick={() => setMenuOpen(false)}>Use</a>
              <a href="#benchmark" onClick={() => setMenuOpen(false)}>Benchmark</a>
              <a href={`${REPOSITORY}/tree/main/docs`} target="_blank" rel="noreferrer">Docs</a>
            </div>
            <div className="resource-buttons">
              <a className="btn" href={REPOSITORY} target="_blank" rel="noreferrer">⌘ Code</a>
              <a className="btn" href={`${REPOSITORY}/tree/main/datasets`} target="_blank" rel="noreferrer">▦ Data</a>
              <a className="btn" href="#cite">Paper</a>
            </div>
          </nav>
        </div>
      </header>

      <div id="top" aria-hidden="true" />

      <section className="hero">
        <div className="container">
          <div className="venue-tag">Short-video recommendation benchmark</div>
          <h1 className="hero-title">
            <span className="depth-icon" aria-hidden="true">▥</span>
            Content Depth <em>Matters</em><br />
            <span>in Short-Video Recommendation</span>
          </h1>
          <p className="hero-tagline">
            Rethinking the attention economy through a benchmark that measures not only what people click, but how deeply recommended content develops information.
          </p>
          <p className="project-byline">
            <a href="https://liweidengdavid.github.io/" target="_blank" rel="noreferrer">Liwei Deng<sup>1</sup></a>
            <span>·</span>
            <a href="https://scholar.google.com.au/citations?hl=en&amp;user=XFtCe08AAAAJ" target="_blank" rel="noreferrer">Jing Jiang<sup>1</sup></a>
            <span>·</span>
            <a href="https://zhw.li/" target="_blank" rel="noreferrer">Zhiwei Li<sup>1</sup></a>
            <span>·</span>
            <a href="https://openreview.net/profile?id=~Yang_Wang134" target="_blank" rel="noreferrer">Yang Wang<sup>2</sup></a>
            <span>·</span>
            <a href="https://scholar.google.com.au/citations?user=Pl8m7hMAAAAJ&amp;hl=en" target="_blank" rel="noreferrer">Guodong Long<sup>1</sup></a>
          </p>
          <div className="project-affiliations">
            <p><sup>1</sup> Australian Artificial Intelligence Institute, University of Technology Sydney</p>
            <p><sup>2</sup> Evidence and Research, Department of Health, Disability and Ageing</p>
          </div>
          <div className="uts-logo-wrap">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="./branding/uts-logo-wide.png" alt="University of Technology Sydney" />
          </div>
          <div className="hero-links">
            <a className="btn hero-btn" href={REPOSITORY} target="_blank" rel="noreferrer">⌘ Code</a>
            <a className="btn hero-btn" href={`${REPOSITORY}/tree/main/datasets`} target="_blank" rel="noreferrer">▦ Dataset</a>
            <a className="btn hero-btn" href="#use">▶ Get started</a>
            <span className="btn hero-btn btn-disabled">Paper soon</span>
          </div>
        </div>
      </section>

      <section className="metric-section">
        <div className="container">
          <div className="hero-metrics">
            <div className="hero-metric-card"><strong>150K+</strong><span>short videos in the full benchmark collection</span></div>
            <div className="hero-metric-card"><strong>1M+</strong><span>interactions in the full source dataset</span></div>
            <div className="hero-metric-card"><strong>13</strong><span>representative paper-facing baselines</span></div>
            <div className="hero-metric-card"><strong>54</strong><span>model implementations across five settings</span></div>
          </div>
          <p className="metric-note">SCOPE-Bench is open to researchers working on recommendation beyond engagement-only objectives.</p>
        </div>
      </section>

      <section className="section bg-soft" id="abstract">
        <div className="container narrow-heading">
          <h2 className="section-title">Abstract</h2>
        </div>
        <div className="container">
          <div className="abstract-box">
            <p>
              Driven by the <span className="highlight-pill economy">attention economy</span>, short-video Recommender Systems (RSs) are primarily optimized to maximize user engagement by promoting videos that capture attention within seconds. These systems inherently favor <span className="highlight-pill shallow">shallow-content videos</span> that are effective at attracting immediate attention. However, growing evidence suggests that prolonged exposure to such content may negatively affect users&apos; <span className="highlight-pill cognition">cognitive engagement and mental well-being</span>, raising concerns about the long-term societal impact of the short-video platform.
            </p>
            <p>
              To tackle this challenge, this paper introduces a new metric, the <span className="highlight-pill cds">Content Depth Score (CDS)</span>, to quantify the content depth of short videos. CDS measures the extent to which a video is expected to stimulate higher-order cognitive processes, using a <span className="highlight-pill scale">seven-level scale</span> grounded in established theories of cognitive psychology and learning. As an initial step toward this vision, we present <span className="highlight-pill benchmark">SCOPE-Bench</span>, the first benchmark for content-depth evaluation in short-video recommendation. Built upon a large-scale open-source short-video dataset, SCOPE-Bench provides CDS annotations for <span className="highlight-pill data">150K videos</span>, enabling systematic evaluation of RSs from a cognitive-content perspective.
            </p>
            <p>
              Leveraging SCOPE-Bench, we evaluate <span className="highlight-pill baselines">13 representative RSs</span> and reveal a consistent preference for shallow-content videos. Moreover, we find that these algorithms recommending cognitively deep content are <span className="highlight-pill finding">only marginally better than random selection</span>, highlighting a previously overlooked limitation of existing recommendation objectives.
            </p>
          </div>
        </div>
      </section>

      <section className="section entry-section" id="use">
        <div className="container">
          <div className="center-heading">
            <h2 className="section-title">Three ways in</h2>
            <p>SCOPE-Bench is a paper project, a runnable benchmark and an extensible model library. Pick the path that matches what you need.</p>
          </div>
          <div className="entry-grid">
            <a className="entry-card" href="#quickstart">
              <span>For builders</span><strong>Run SCOPE-Bench</strong><p>Install the benchmark and evaluate one model with unified relevance and depth metrics.</p><b>→ Get started</b>
            </a>
            <a className="entry-card" href="#benchmark">
              <span>For understanding</span><strong>Explore the benchmark</strong><p>See how CDS and LCDS connect video-level content analysis to top-k recommendation.</p><b>→ Read the overview</b>
            </a>
            <a className="entry-card" href={`${REPOSITORY}/blob/main/docs/models.md`} target="_blank" rel="noreferrer">
              <span>For comparison</span><strong>Browse the models</strong><p>Centralized, multimodal, federated and sequential implementations in one registry.</p><b>→ View model coverage</b>
            </a>
          </div>
        </div>
      </section>

      <section className="section" id="benchmark">
        <div className="container">
          <div className="question-heading">
            <span>①</span>
            <div><h2>What depth does a video contain?</h2><p>Item-level evaluation separates surface-level presentation from content that develops information, context and reasoning.</p></div>
          </div>
          <figure className="paper-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="./figures/overview-a.png" alt="SCOPE-Bench item-level Content Depth Score annotation workflow" />
            <figcaption>Item-level CDS annotation workflow. CDS captures informational and reasoning depth; factual correctness and safety remain independent dimensions.</figcaption>
          </figure>
        </div>
      </section>

      <section className="section bg-soft">
        <div className="container">
          <div className="question-heading">
            <span>②</span>
            <div><h2>What depth does a recommender deliver?</h2><p>List-level evaluation measures the depth profile of a complete top-k recommendation list without discarding conventional ranking quality.</p></div>
          </div>
          <figure className="paper-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="./figures/overview-b.png" alt="SCOPE-Bench list-level cognitive-aware recommender evaluation workflow" />
            <figcaption>Recommendation-list-level cognitive-aware evaluation, reporting Recall, NDCG, Precision, A-LCDS and E-LCDS together.</figcaption>
          </figure>
        </div>
      </section>

      <section className="section mechanism-section">
        <div className="container">
          <div className="center-heading">
            <h2 className="section-title">One benchmark, five settings</h2>
            <p>The repository brings distinct recommendation families into a shared training and evaluation contract.</p>
          </div>
          <div className="mechanism-grid">
            <article className="mechanism amber"><span>01</span><h3>Centralized ID</h3><p>Classical, graph-based, diffusion and variational recommendation models.</p></article>
            <article className="mechanism violet"><span>02</span><h3>Multimodal</h3><p>Factorization, contrastive and graph approaches built around content features.</p></article>
            <article className="mechanism teal"><span>03</span><h3>Federated</h3><p>ID-only and multimodal methods for distributed recommendation research.</p></article>
            <article className="mechanism rust"><span>04</span><h3>Sequential</h3><p>Sequence-aware baselines connected to the same evaluation interface.</p></article>
          </div>
        </div>
      </section>

      <section className="section bg-soft" id="quickstart">
        <div className="container quickstart-layout">
          <div>
            <span className="section-number">③</span>
            <h2 className="section-title left">From clone to benchmark</h2>
            <p className="section-lede">Select a model and dataset, then receive engagement and content-depth metrics in the same result files.</p>
            <a className="text-link" href={`${REPOSITORY}/blob/main/docs/Tutorial/01-quick-start.md`} target="_blank" rel="noreferrer">Read the complete tutorial →</a>
          </div>
          <div className="code-card">
            <div><span>QUICK START</span><span>v1.0</span></div>
            <pre><code><i># install</i>{"\n"}$ git clone {REPOSITORY}.git{"\n"}$ cd SCOPE-Bench{"\n"}$ pip install -e <b>&quot;.[torch,multimodal,hpo]&quot;</b>{"\n\n"}<i># train and evaluate</i>{"\n"}$ python main.py \\{"\n"}    --model <b>LightGCN</b> \\{"\n"}    --dataset <b>ShortVideoSampled</b> \\{"\n"}    --gpu_id <b>0</b></code></pre>
          </div>
        </div>
      </section>

      <section className="section analysis-section">
        <div className="container">
          <div className="center-heading"><h2 className="section-title">A living research project</h2></div>
          <div className="future-callout">
            <span aria-hidden="true">↗</span>
            <div><strong>More work is on the way.</strong><p>SCOPE-Bench will continue to grow with new research, artifacts and evaluation capabilities. Follow the repository for future updates.</p></div>
            <a className="btn" href={REPOSITORY} target="_blank" rel="noreferrer">Watch on GitHub</a>
          </div>
        </div>
      </section>

      <section className="section cite-section bg-soft" id="cite">
        <div className="container">
          <h2 className="section-title">Coming soon</h2>
          <div className="citation-coming">
            <span aria-hidden="true">⌛</span>
            <p>The paper and official citation details will be released here.</p>
          </div>
        </div>
      </section>

      <footer>
        <div className="container footer-inner">
          <span><strong>SCOPE-Bench</strong> · Content depth matters.</span>
          <span>v1.0 · <a href={REPOSITORY}>Code</a> · <a href={`${REPOSITORY}/tree/main/docs`}>Docs</a> · <a href={`${REPOSITORY}/blob/main/CONTRIBUTING.md`}>Contribute</a></span>
        </div>
      </footer>
    </main>
  );
}
