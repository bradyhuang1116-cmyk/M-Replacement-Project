import os from "node:os";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  devIndicators: false,
  // Next.js dev allowlist does not support a bare "*" wildcard.
  // Use the broadest supported patterns for LAN development instead.
  allowedDevOrigins: ["*.*.*.*", os.hostname()],
};

export default nextConfig;
