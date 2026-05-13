# GlowCart — Setup & Run Instructions

## Prerequisites

| Tool       | Minimum Version | Tested On |
|------------|-----------------|-----------|
| Python     | 3.11+           | 3.13      |
| Node.js    | 18+             | 22        |
| Docker     | 20+             | (optional)|

---

## Option 1: Run Locally

### Step 1 — Install Backend Dependencies

```bash
cd milestone2/core
pip install -r requirements.txt
```

### Step 2 — Start the Flask Backend

```bash
python app.py
```

The API server will start on **http://localhost:8000**.

### Step 3 — Install Frontend Dependencies

Open a **separate terminal** and run:

```bash
cd milestone2/web
npm install
```

### Step 4 — Start the React Development Server

```bash
npm run dev
```

The frontend will start on **http://localhost:5173** and automatically proxy API requests to the backend on port 8000.

### Step 5 — Open the Application

Navigate to **http://localhost:5173** in your browser.

---

## Option 2: Run with Docker

The project includes a multi-stage Dockerfile that builds both the frontend and backend into a single container.

### Step 1 — Build the Docker Image

```bash
cd milestone2
docker build -t glowcart .
```

### Step 2 — Run the Container

```bash
docker run -p 8000:8000 glowcart
```

### Step 3 — Open the Application

Navigate to **http://localhost:8000** in your browser.

> **Note:** When running via Docker, the built React frontend is served directly by Flask as static files, so only port 8000 is needed.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install` fails on native packages | Ensure `gcc` / `g++` are installed (required by some Python dependencies). |
| Port 8000 already in use | Stop the conflicting process or change the port with `PORT=9000 python app.py`. |
| Frontend cannot reach the API | Verify that the Flask backend is running on port 8000 before starting the frontend dev server. |
| Docker build fails at npm step | Ensure you are running `docker build` from the `milestone2/` directory. |
