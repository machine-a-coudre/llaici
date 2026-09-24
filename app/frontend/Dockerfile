FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

# Overwritten by a bind mount in docker-compose.yml for dev (Vite's HMR picks up
# changes automatically); this COPY just makes the image runnable standalone too.
COPY . .

EXPOSE 5173

# vite.config.ts sets server.host = true (binds 0.0.0.0), needed for the
# container's port mapping to reach it — Vite defaults to localhost-only.
CMD ["npm", "run", "dev"]
