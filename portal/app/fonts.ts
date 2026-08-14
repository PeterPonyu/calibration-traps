import localFont from "next/font/local";

// Self-hosted type pair (OFL 1.1, see app/fonts/OFL-*.txt). Build performs no
// font fetch. Sources: @fontsource/jetbrains-mono + @fontsource/newsreader.
export const jetbrainsMono = localFont({
  src: [
    { path: "./fonts/jetbrains-mono-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/jetbrains-mono-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/jetbrains-mono-latin-600-normal.woff2", weight: "600", style: "normal" },
    { path: "./fonts/jetbrains-mono-latin-700-normal.woff2", weight: "700", style: "normal" },
  ],
  variable: "--font-chrome",
  display: "swap",
});

export const newsreader = localFont({
  src: [
    { path: "./fonts/newsreader-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/newsreader-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/newsreader-latin-600-normal.woff2", weight: "600", style: "normal" },
    { path: "./fonts/newsreader-latin-400-italic.woff2", weight: "400", style: "italic" },
    { path: "./fonts/newsreader-latin-500-italic.woff2", weight: "500", style: "italic" },
    { path: "./fonts/newsreader-latin-600-italic.woff2", weight: "600", style: "italic" },
  ],
  variable: "--font-prose",
  display: "swap",
});
