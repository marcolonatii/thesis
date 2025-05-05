# File: demo/frontend/frontend.Dockerfile

# --- Base Stage ---
# Use your preferred Node.js base image (sticking with yours)
FROM node:22.9.0 AS base

# Set working directory
WORKDIR /app

# Copy package.json and yarn.lock first for dependency caching
COPY package.json ./
COPY yarn.lock ./

# Install dependencies using yarn
RUN yarn install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# --- Development Stage ---
# Inherit from the base stage
FROM base AS development

# Set Node environment to development
ENV NODE_ENV=development

# Expose the default React development port (usually 3000 for CRA/Vite)
# Double-check your package.json start script if it uses a different port.
EXPOSE 5173

# Default command to start the development server (using yarn)
# Check your package.json "scripts" section for the correct command (e.g., "start", "dev")
CMD ["yarn", "dev", "--host"]


# --- Build Stage (for Production) ---
# Inherit from the base stage
FROM base AS build

# Set Node environment to production
ENV NODE_ENV=production

# Accept build-time arguments (e.g., API URL)
ARG REACT_APP_API_URL=http://localhost:7263
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

# Build the application using yarn
RUN yarn build
# Build output seems to be /app/dist based on your original file


# --- Production Stage ---
# Use your preferred Nginx base image
FROM nginx:latest AS production

# Copy the built static files from the build stage's output directory (/app/dist)
COPY --from=build /app/dist /usr/share/nginx/html

# Expose the default Nginx port
EXPOSE 80

# Default command to run Nginx
CMD ["nginx", "-g", "daemon off;"]


# --- Default Stage ---
# If `docker build` is run without --target, it will build the 'production' stage.
# If you primarily run for development, you could change this default:
# FROM development
# But defaulting to production is usually safer.
FROM production