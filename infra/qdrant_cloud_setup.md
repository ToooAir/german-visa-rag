# Qdrant Cloud Setup Guide

For the production deployment on GCP Cloud Run, we use [Qdrant Cloud](https://cloud.qdrant.io/) as the managed vector database.

## Steps:
1. Create an account at `cloud.qdrant.io`.
2. Create a new Cluster (The "Free Tier" 1GB RAM / 10GB storage is enough).
3. Generate an API Key in the Data Access section.
4. Copy the Cluster URL and API Key.
5. Add them to your GCP Secret Manager:
   - `qdrant-url`: `https://your-cluster-id.region.gcp.cloud.qdrant.io`
   - `qdrant-api-key`: `your-generated-api-key`
