# Phase 6 — Deployment Inventory

## Purpose

This document defines the files, runtime dependencies, configuration, secrets, and generated artifacts required to deploy the patent similarity application.

Phase 2 retrieval artifacts are treated as **static production artifacts**. They are not rebuilt or re-embedded during application startup.

---

## 1. Application Components

The production application consists of:

* **Phase 2 — Embedding + Retrieval**

  * Gemini embedding client
  * FAISS retrieval index
  * FAISS metadata
  * patent database
* **Phase 3 — Ranking + Synthesis**

  * Gemini LLM synthesis
  * similarity scoring
  * evidence generation
* **Phase 4 — Backend/API**

  * FastAPI application
  * HTTP API
  * application startup/lifecycle
* **Phase 5 — Frontend**

  * User interface
  * backend API integration

Phase 4 and Phase 5 deployment commands and interface details remain subject to the final integration provided by the project partner.

---

## 2. Static Production Artifacts

The following artifacts are required at deployment time.

### FAISS index

Path:

```text
embeddings/faiss_index/patent_similarity.faiss
```

Expected properties:

* Index type: `IndexFlatIP`
* Vector dimension: `3072`
* Vector count: `6779`

SHA256:

```text
D5B4B5CED58E0FD8D8EEF97311058F7C31C24CC8B14BCC8AB531E8677A9C2911
```

### FAISS metadata

Path:

```text
embeddings/faiss_index/patent_similarity_metadata.json
```

Expected properties:

* Contains metadata corresponding to the FAISS vectors
* Must remain aligned with the FAISS index
* Expected metadata/vector count: `6779`

SHA256:

```text
5D3F3513EF72CF4418A16494C54E7D1132B7A56F0FA97A2CC5DE6EFD099ED0E7
```

### Patent database

Path:

```text
data/patents.db
```

Expected production database:

* SQLite
* `patents` table
* corpus contains `6980` patents
* `6779` patents have abstracts and therefore corresponding FAISS vectors

The database is read at application runtime and is not regenerated during deployment.

---

## 3. Runtime Source Code

The deployment must include the application source required by:

```text
src/phase2_embedding_retrieval/
src/phase3_synthesis/
```

and the final Phase 4/Phase 5 application code.

The deployment must preserve the existing Phase 2 retrieval implementation and its compatibility with the production FAISS artifacts.

---

## 4. Python Dependencies

The production environment must install the dependencies required by:

* Phase 2 embedding/retrieval
* Phase 3 synthesis
* Phase 4 FastAPI backend
* Phase 5 frontend

The final dependency list should be consolidated into the project's production dependency file after Phase 4/5 integration.

Known required packages from the current Phase 2/3 implementation include dependencies providing:

* Google GenAI SDK
* FAISS
* NumPy
* python-dotenv

Additional dependencies may be required by FastAPI and the final frontend/backend integration.

---

## 5. Secrets and Environment Variables

### Required secret

```text
GEMINI_API_KEY
```

The current Phase 2 authentication flow:

```text
load_dotenv()
        ↓
os.getenv("GEMINI_API_KEY")
        ↓
genai.Client(api_key=api_key)
```

Phase 3 reuses the same Phase 2 `get_client()` implementation.

Therefore the deployment environment must provide:

```text
GEMINI_API_KEY=<production-secret>
```

The API key must **not** be committed to Git or included in the Docker image.

No `GOOGLE_API_KEY` variable is used by the current Phase 2/3 implementation.

---

## 6. Generated at Runtime

The following should be generated or initialized by the application/runtime rather than committed as new deployment artifacts:

* Gemini API client/session state
* temporary request data
* application logs
* temporary files
* process/runtime state
* framework-generated caches, where applicable

The production application must **not** rebuild the FAISS index during startup.

---

## 7. Development/Evaluation Artifacts

The following are useful for development and evaluation but are not required for the production service itself:

```text
data/evaluation_cases.json
data/evaluation_report.json
data/full_corpus_eval_results.json
data/no_match_eval_results.json
data/positive_rankings.json
data/stage2_evaluation_results.json
```

Evaluation scripts under:

```text
test/
```

are likewise not required for the runtime container unless explicitly included for deployment smoke tests.

---

## 8. Files That Must Never Contain Production Secrets

The following must not contain a real Gemini API key:

```text
.git/
.env.example
Dockerfile
docker-compose.yml
source code
documentation
logs committed to Git
```

A local `.env` file may contain the developer's key but must remain ignored by Git.

---

## 9. Production Artifact Validation

Before deployment, the following must be verified automatically:

| Check                     |         Expected |
| ------------------------- | ---------------: |
| FAISS index exists        |              Yes |
| FAISS dimension           |             3072 |
| FAISS vector count        |             6779 |
| FAISS SHA256              |  `D5B4...9C2911` |
| Metadata exists           |              Yes |
| Metadata count            |             6779 |
| Metadata SHA256           | `5D3F...99ED0E7` |
| Patent database exists    |              Yes |
| Gemini API key configured |              Yes |
| FAISS/metadata alignment  |              Yes |

These checks will be implemented in:

```text
scripts/verify_deployment_artifacts.py
```

The verification script should be runnable before deployment and should fail with a non-zero exit code if a required artifact is missing, corrupted, or structurally inconsistent.

---

## 10. Deployment Principles

### Static retrieval artifacts

FAISS index and metadata are versioned production inputs.

They should be:

1. generated offline,
2. validated,
3. checksummed,
4. shipped with the application,
5. loaded at startup.

They should **not** be rebuilt in the production process.

### Secrets

Secrets are supplied through the deployment platform's environment/secret configuration.

They are not stored in source control.

### Release integrity

The application code, FAISS index, metadata, and compatible database should be treated as a release unit.

A future corpus/index refresh should produce a new validated artifact set rather than modifying the live index in place.

### Phase boundaries

Phase 6 should not modify the frozen Phase 2 retrieval implementation merely to make deployment work.

Any deployment-specific changes should be isolated to deployment configuration or the Phase 4/5 integration layer.

---

## 11. Currently Confirmed Deployment Facts

The following have been independently verified:

* FAISS vector dimension: **3072**
* FAISS vector count: **6779**
* FAISS index filename: `patent_similarity.faiss`
* FAISS metadata filename: `patent_similarity_metadata.json`
* FAISS SHA256: verified
* Metadata SHA256: verified
* FAISS/metadata alignment: verified
* Gemini authentication variable: `GEMINI_API_KEY`
* Phase 3 reuses the Phase 2 Gemini client configuration

The remaining deployment-specific items depend on the final Phase 4/5 integration, including:

* FastAPI startup command
* API request/response contract
* backend port
* frontend build/start command
* frontend environment variables
* frontend-to-backend URL configuration
* final production hosting configuration
