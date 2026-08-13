# SCOPE-Bench project website

This directory contains the static academic project website for SCOPE-Bench.

## Local preview

```bash
npm install
npm run dev
```

## Production builds

```bash
npm run build        # local/portable static build
npm run build:pages  # GitHub Pages asset paths
npm test
```

Pushing website changes to `main` triggers `.github/workflows/pages.yml` and deploys `dist/client` through GitHub Pages.
