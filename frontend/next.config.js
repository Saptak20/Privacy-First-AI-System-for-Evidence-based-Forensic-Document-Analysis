/** @type {import('next').NextConfig} */
const nextConfig = {
    /* proxy API requests to the FastAPI backend during development */
    async rewrites() {
        return [
            {
                source: "/api/:path*",
                destination: "http://127.0.0.1:8000/api/:path*",
            },
        ];
    },
};

module.exports = nextConfig;
