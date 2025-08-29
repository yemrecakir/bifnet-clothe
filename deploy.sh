#!/bin/bash

# Google Cloud Deployment Script for BiRefNet API

echo "🌈 BiRefNet API - Google Cloud Deployment"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found!"
    echo "💡 Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set variables
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME="birefnet-api"

echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Service: $SERVICE_NAME"
echo ""

# Authenticate and set project
echo "🔐 Setting up Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "⚡ Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy
echo "🚀 Starting deployment..."
gcloud builds submit --config cloudbuild.yaml .

echo "✅ Deployment complete!"
echo "🌐 Your API is available at:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'

echo ""
echo "🎯 Test your deployed API:"
echo "curl \$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')/health"