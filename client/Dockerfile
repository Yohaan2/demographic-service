# ==============================================================================
# STAGE 1: Node.js static builder
# ==============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy client source code and compile Vite distribution
COPY . .
RUN npm run build


# ==============================================================================
# STAGE 2: Lightweight Nginx server
# ==============================================================================
FROM nginx:stable-alpine AS runner

# Copy built assets to Nginx static serving directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Run Nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]
