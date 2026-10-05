Deployment notes and recommended cloud-friendly structure

1) Build artifacts
- Dockerfile included to build a container image for the FastAPI app.
- docker-compose.yml for local dev and quick testing.

2) CI / CD
- Recommended: GitHub Actions workflow to build and push a container to GitHub Container Registry (ghcr.io) or ECR.
- Example steps:
  - checkout
  - build image
  - push to registry
  - optionally deploy to ECS/Fargate or EKS

3) AWS deployment options
- Small demo: ECS/Fargate with Application Load Balancer, RDS or S3 for data, and Secrets Manager for API keys.
- Production: EKS with autoscaling and managed Postgres / S3 data lake / SageMaker for models.

4) Secrets
- Do NOT store API keys in code. Use GitHub Actions secrets, AWS Secrets Manager, or environment variables.

5) Next infra additions I can scaffold for you
- GitHub Actions workflow to build and push Docker image.
- Terraform skeleton for ECS + ALB + IAM roles.
- CloudFormation / CDK stack for a full demo environment.
