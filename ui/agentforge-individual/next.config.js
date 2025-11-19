/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    turbo: {
      root: '/Users/baileymahoney/AgentForge/ui/agentforge-user'
    }
  },
  transpilePackages: ['lucide-react', 'framer-motion', 'valtio'],
  typescript: {
    ignoreBuildErrors: false
  },
  eslint: {
    ignoreDuringBuilds: false
  }
};

module.exports = nextConfig;

