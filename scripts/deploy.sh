#!/bin/bash
# Deployment script for GCP Cloud Run (Service & Job)

set -e

usage() {
    echo "Usage: $0 [-e ENVIRONMENT] [-p PROJECT_ID] [-r REGION]"
    echo "  -e ENVIRONMENT    : staging or production (default: staging)"
    echo "  -p PROJECT_ID     : GCP Project ID"
    echo "  -r REGION         : GCP Region (default: europe-west1)"
    exit 1
}

ENVIRONMENT="staging"
REGION="europe-west1"
SERVICE_NAME="german-visa-rag"

while getopts "e:p:r:" opt; do
    case $opt in
        e) ENVIRONMENT=$OPTARG ;;
        p) PROJECT_ID=$OPTARG ;;
        r) REGION=$OPTARG ;;
        *) usage ;;
    esac
done

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP Project ID required (-p flag)"
    usage
fi

echo "================================================"
echo "🇩🇪 German Visa RAG - Deployment"
echo "================================================"
echo "Environment: $ENVIRONMENT"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Validate environment
if [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    echo "❌ Error: Invalid environment. Must be 'staging' or 'production'"
    exit 1
fi

# Load environment variables
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    exit 1
fi
source .env

echo "📦 1. Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:$ENVIRONMENT .
docker tag gcr.io/$PROJECT_ID/$SERVICE_NAME:$ENVIRONMENT gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

echo "🔐 2. Authenticating with GCP..."
gcloud auth configure-docker gcr.io --quiet

echo "📤 3. Pushing image to Container Registry..."
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:$ENVIRONMENT
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

SERVICE_FULL_NAME="$SERVICE_NAME-api-$ENVIRONMENT"
JOB_FULL_NAME="$SERVICE_NAME-job-$ENVIRONMENT"

echo "🚀 4. Deploying Web API (Cloud Run Service)..."
gcloud run deploy $SERVICE_FULL_NAME \
    --image=gcr.io/$PROJECT_ID/$SERVICE_NAME:$ENVIRONMENT \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=$([ "$ENVIRONMENT" == "production" ] && echo "4Gi" || echo "2Gi") \
    --cpu=$([ "$ENVIRONMENT" == "production" ] && echo "4" || echo "2") \
    --timeout=600 \
    --set-env-vars=ENVIRONMENT=$ENVIRONMENT,USE_OLLAMA=false \
    --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
    --set-secrets=QDRANT_URL=qdrant-url:latest \
    --set-secrets=QDRANT_API_KEY=qdrant-api-key:latest \
    --set-secrets=REDIS_URL=redis-url:latest \
    --set-secrets=API_KEY=api-key:latest \
    --project=$PROJECT_ID

echo "⚙️ 5. Deploying Ingestion Pipeline (Cloud Run Job)..."
gcloud run jobs deploy $JOB_FULL_NAME \
    --image=gcr.io/$PROJECT_ID/$SERVICE_NAME:$ENVIRONMENT \
    --region=$REGION \
    --command="python" \
    --args="-m,src.ingestion.cli,ingest" \
    --memory=2Gi \
    --cpu=2 \
    --task-timeout=3600 \
    --set-env-vars=ENVIRONMENT=$ENVIRONMENT,USE_OLLAMA=false \
    --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
    --set-secrets=QDRANT_URL=qdrant-url:latest \
    --set-secrets=QDRANT_API_KEY=qdrant-api-key:latest \
    --project=$PROJECT_ID

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Getting service URL..."
CLOUD_RUN_URL=$(gcloud run services describe $SERVICE_FULL_NAME \
    --platform=managed \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format='value(status.url)')

echo ""
echo "🌐 API Service URL: $CLOUD_RUN_URL"
echo ""
echo "Testing service..."
curl -f $CLOUD_RUN_URL/v1/health || echo "⚠️  Health check failed"

echo ""
echo "📝 Next steps:"
echo "  1. View logs: gcloud run logs read $SERVICE_FULL_NAME --region=$REGION"
echo "  2. View service: gcloud run services describe $SERVICE_FULL_NAME --region=$REGION"
echo "  3. Trigger Crawler: gcloud run jobs execute $JOB_FULL_NAME --region=$REGION"
echo "  4. Test endpoint: curl -H 'X-API-Key: $API_KEY' $CLOUD_RUN_URL/v1/health"
