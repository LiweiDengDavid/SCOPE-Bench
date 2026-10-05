import type { Metadata } from "next";
import { Crimson_Pro, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const serif = Crimson_Pro({
  variable: "--font-serif",
  subsets: ["latin"],
});

const sans = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://liweidengdavid.github.io/SCOPE-Bench/"),
  title: "SCOPE-Bench — Content Depth Matters",
  description:
    "A living benchmark for evaluating content depth in short-video recommendation, from individual videos to top-k recommendation lists.",
  keywords: [
    "SCOPE-Bench",
    "short-video recommendation",
    "content depth",
    "recommender systems",
    "benchmark",
  ],
  authors: [
    { name: "Liwei Deng" },
    { name: "Jing Jiang" },
    { name: "Zhiwei Li" },
    { name: "Allison Clarke" },
    { name: "Yang Wang" },
    { name: "Guodong Long" },
  ],
  icons: {
    icon: "./favicon.svg",
    shortcut: "./favicon.svg",
  },
  openGraph: {
    title: "SCOPE-Bench — Content Depth Matters",
    description:
      "A living benchmark for content-depth-aware short-video recommendation.",
    type: "website",
    images: [{ url: "og.png", width: 1200, height: 630, alt: "SCOPE-Bench" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SCOPE-Bench — Content Depth Matters",
    description:
      "A living benchmark for content-depth-aware short-video recommendation.",
    images: ["og.png"],
  },
  other: {
    citation_title:
      "Content Depth Matters in Short-Video Recommendation: Rethinking the Attention Economy",
    citation_author: [
      "Liwei Deng",
      "Jing Jiang",
      "Zhiwei Li",
      "Yang Wang",
      "Guodong Long",
    ],
    citation_publication_date: "2026/08/14",
    citation_arxiv_id: "2608.13990",
    citation_pdf_url: "https://arxiv.org/pdf/2608.13990",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
        {children}
      </body>
    </html>
  );
}
