/** @type {import('next').NextConfig} */
const apiUrl = process.env.FOXMESH_API_URL || "http://127.0.0.1:8000";

const nextConfig = {
  // The browser talks to /api on this origin; Next proxies it to the FastAPI backend.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
};

export default nextConfig;
