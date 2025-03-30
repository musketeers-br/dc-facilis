 [![Gitter](https://img.shields.io/badge/Available%20on-Intersystems%20Open%20Exchange-00b2a9.svg)](https://openexchange.intersystems.com/package/Facilis)
 [![Quality Gate Status](https://community.objectscriptquality.com/api/project_badges/measure?project=intersystems_iris_community%2Fintersystems-iris-dev-template&metric=alert_status)](https://community.objectscriptquality.com/dashboard?id=intersystems_iris_community%2Fintersystems-iris-dev-template)
 [![Reliability Rating](https://community.objectscriptquality.com/api/project_badges/measure?project=intersystems_iris_community%2Fintersystems-iris-dev-template&metric=reliability_rating)](https://community.objectscriptquality.com/dashboard?id=intersystems_iris_community%2Fintersystems-iris-dev-template)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat&logo=AdGuard)](LICENSE)

# Facilis - Effortless API Interoperability with AI

![Facilis made by AI](./facilis.jpeg)	

Facilis is an AI-powered solution designed to streamline API integration by extracting structured specifications from natural language descriptions. Now powered by CrewAI, Facilis ensures a more efficient and modular approach to handling API documentation and interoperability.

**How to pronounce Facilis**: [ˈfäkɪlʲɪs̠]

## 🚀 Motivation
Managing API integrations can be complex and error-prone. Facilis simplifies this process by leveraging AI to extract and validate API details, ensuring compliance with OpenAPI standards and facilitating seamless interoperability.

## 🛠️ How It Works
Facilis processes user-provided API descriptions, extracting key details such as endpoints, HTTP methods, parameters, authentication, and more. It then structures the extracted information into OpenAPI-compliant JSON and exports it to InterSystems IRIS for interoperability. The workflow consists of multiple AI agents, orchestrated by CrewAI, handling:

- **Extraction**: Identifies and structures API specifications from natural language.
- **Interaction**: Requests missing details from the user.
- **Validation**: Ensures API compliance and consistency.
- **Transformation**: Converts structured API details into OpenAPI JSON.
- **Review**: Verifies OpenAPI compliance before finalization.
- **Export**: Converts and exports OpenAPI documentation to InterSystems IRIS.

## 📋 Prerequisites
- Python 3.9+
- Docker & Docker Compose
- An API key from your preferred LLM provider (OpenAI, Azure OpenAI, Google Gemini, or Claude)

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/musketeers-br/dc-facilis
cd dc-facilis
```

### 2️⃣ Set Up Environment Variables
Facilis requires **two** `.env` files, one in the root directory and another in `python/facilis/`.

#### Create `.env` files based on the provided samples
```bash
cp env_sample .env
cp python/facilis/env_sample python/facilis/.env
```

#### Edit `.env` and `python/facilis/.env` with your settings
```ini
# AI Engine selection (choose one: openai, azureopenai, googleGemini, claude, ollama)
AI_ENGINE=openai  

# LLM Model Name
LLM_MODEL_NAME=gpt-4o-mini

# API Key for the selected AI provider
OPENAI_API_KEY=your-api-key-here
```

### 3️⃣ Run with Docker Compose
```bash
# Build the container
docker-compose build --no-cache --progress=plain

# Start the application
docker-compose up -d

# Stop and remove containers
docker-compose down --rmi all
```

## 🚧 Limitations
- Accuracy of extracted API details depends on the quality of user input.
- LLM-based processing may introduce minor inconsistencies requiring manual review.
- Currently optimized for OpenAPI 3.0.

## 🎖️ Credits
Facilis is developed with ❤️ by the Musketeers Team

* [José Roberto Pereira](https://community.intersystems.com/user/jos%C3%A9-roberto-pereira-0)
* [Henry Pereira](https://community.intersystems.com/user/henry-pereira)
* [Henrique Dias](https://community.intersystems.com/user/henrique-dias-2)
