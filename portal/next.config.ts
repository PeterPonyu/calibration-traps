import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/calibration-traps",
  trailingSlash: true,
  images: { unoptimized: true },
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
