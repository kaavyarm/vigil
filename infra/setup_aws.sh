#!/usr/bin/env bash
# AWS infrastructure setup for Vigil.
# Run once from a machine with AWS CLI v2 configured (aws configure).
# Replace the variables in the block below before running.
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="vigil"
ECS_CLUSTER="vigil-cluster"
ECS_SERVICE="vigil-api"
MODEL_BUCKET="vigil-models-${ACCOUNT_ID}"
DB_INSTANCE="vigil-db"
DB_NAME="vigil"
DB_USER="vigil"
DB_PASSWORD="$(openssl rand -base64 24)"

echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"

# ── S3 bucket for model artifacts ─────────────────────────────────────────────
echo "==> Creating S3 bucket ${MODEL_BUCKET}"
aws s3api create-bucket \
  --bucket "${MODEL_BUCKET}" \
  --region "${REGION}" \
  --create-bucket-configuration LocationConstraint="${REGION}" 2>/dev/null || true

aws s3api put-public-access-block \
  --bucket "${MODEL_BUCKET}" \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo "==> Uploading model artifacts"
aws s3 sync models/ "s3://${MODEL_BUCKET}/models/" \
  --exclude "plots/*" --exclude "*.png"

# ── ECR repository ────────────────────────────────────────────────────────────
echo "==> Creating ECR repository"
aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" 2>/dev/null || true

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "==> Building and pushing Docker image"
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"
docker build -t "${ECR_REPO}:latest" .
docker tag "${ECR_REPO}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"

# ── RDS PostgreSQL (free-tier db.t3.micro) ────────────────────────────────────
echo "==> Creating RDS instance (this takes ~5 minutes)"
aws rds create-db-instance \
  --db-instance-identifier "${DB_INSTANCE}" \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 16 \
  --master-username "${DB_USER}" \
  --master-user-password "${DB_PASSWORD}" \
  --db-name "${DB_NAME}" \
  --allocated-storage 20 \
  --storage-type gp2 \
  --publicly-accessible \
  --no-multi-az \
  --region "${REGION}" 2>/dev/null || true

echo "Waiting for RDS to become available…"
aws rds wait db-instance-available --db-instance-identifier "${DB_INSTANCE}" --region "${REGION}"

DB_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier "${DB_INSTANCE}" \
  --query "DBInstances[0].Endpoint.Address" \
  --output text)

DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_ENDPOINT}:5432/${DB_NAME}"

# ── Secrets Manager ───────────────────────────────────────────────────────────
echo "==> Storing DATABASE_URL in Secrets Manager"
aws secretsmanager create-secret \
  --name "vigil/database-url" \
  --secret-string "${DATABASE_URL}" \
  --region "${REGION}" 2>/dev/null || \
aws secretsmanager put-secret-value \
  --secret-id "vigil/database-url" \
  --secret-string "${DATABASE_URL}" \
  --region "${REGION}"

# ── CloudWatch log group ──────────────────────────────────────────────────────
echo "==> Creating CloudWatch log group"
aws logs create-log-group --log-group-name "/ecs/vigil-api" --region "${REGION}" 2>/dev/null || true

# ── ECS Fargate cluster + service ─────────────────────────────────────────────
echo "==> Creating ECS cluster"
aws ecs create-cluster --cluster-name "${ECS_CLUSTER}" --region "${REGION}" 2>/dev/null || true

echo "==> Registering task definition"
sed -e "s/ACCOUNT_ID/${ACCOUNT_ID}/g" \
    -e "s/REGION/${REGION}/g" \
    infra/ecs_task_definition.json | \
aws ecs register-task-definition --cli-input-json file:///dev/stdin --region "${REGION}"

echo ""
echo "==> Done!"
echo "    DATABASE_URL: ${DATABASE_URL}"
echo "    ECR image:    ${ECR_URI}:latest"
echo "    Next step:    Create ECS service and ALB via the AWS console or:"
echo "    aws ecs create-service --cluster ${ECS_CLUSTER} --service-name ${ECS_SERVICE} \\"
echo "      --task-definition vigil-api --desired-count 1 --launch-type FARGATE \\"
echo "      --network-configuration 'awsvpcConfiguration={subnets=[SUBNET_ID],securityGroups=[SG_ID],assignPublicIp=ENABLED}'"
