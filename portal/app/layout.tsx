import type { Metadata } from "next";
import { jetbrainsMono, newsreader } from "./fonts";
import "./globals.css";

export const metadata: Metadata = {
  title: "Calibration Traps — Preflight Instrument",
  description:
    "Phosphor preflight console for the calibration-traps warehouse. HOLD: not live inference.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${jetbrainsMono.variable} ${newsreader.variable}`}>
        {children}
      </body>
    </html>
  );
}
