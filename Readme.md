---

# 📘 Project Instructions

## ⚙️ Docker Compose Commands

Use the following commands to manage the FastAPI backend using Docker Compose:
---

```bash
# Build the Docker image
docker-compose build

# Start the containers
docker-compose up

# start and build
docker-compose up --build

# Start containers in the background (detached mode)
docker-compose up -d

# Stop the containers
docker-compose down

# View logs
docker-compose logs -f
```

---

## 🔐 OCI Configuration (Oracle Cloud)

To enable sentiment analysis via OCI Language Services:

1. Log in to your [OCI Console](https://cloud.oracle.com/).
2. Generate your OCI **config file** and **API key file**:
   - Follow [Oracle’s instructions](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm#configfile) to set this up.
3. Create a `.oci` folder in the project root (if not already present):
   ```bash
   mkdir -p .oci
   ```
4. Place both the `config` and `oci_api_key.pem` files inside the `.oci` folder.

The structure should look like:
```
project-root/
│
├── .oci/
│   ├── config
│   └── oci_api_key.pem
```

---

## 🔑 Environment Variables

To enable Google Gemini AI features, set up an `.env` file:

1. Generate your **Gemini API Key** from [Google AI Studio](https://makersuite.google.com/).
2. Create a `.env` file in the project root with the following content:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

3. This file will be automatically loaded by the app when using Docker Compose.

---

