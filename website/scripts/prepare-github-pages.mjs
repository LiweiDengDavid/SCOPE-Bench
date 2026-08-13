import { cp } from "node:fs/promises";

const artifactRoot = new URL("../dist/client/", import.meta.url);
const prefixedAssets = new URL("SCOPE-Bench/_next/", artifactRoot);
const publishedAssets = new URL("_next/", artifactRoot);

// vinext writes assetPrefix files under a matching directory. GitHub Pages
// already mounts this artifact at /SCOPE-Bench, so expose a root-level copy.
await cp(prefixedAssets, publishedAssets, { recursive: true });
