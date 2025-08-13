# Deploy Prefect Helm Chart

This Helm chart deploys a Kubernetes Job that runs Prefect flow deployment scripts.

## Chart Structure

```
charts/deploy-prefect/
├── Chart.yaml                 # Chart metadata
├── values.yaml               # Default values
├── templates/                # Kubernetes manifests
│   ├── _helpers.tpl         # Template helpers
│   └── deploy-prefect-job.yaml # Job template
└── README.md                 # This file
```

## Usage

### From GitHub Actions (Recommended)

The chart is automatically deployed in the GitHub workflow with all necessary values:

```yaml
- name: Deploy to Digital Ocean cluster
  run: |
    cd .github/k8s
    helm upgrade --install deploy-code . \
      --set image.repository=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }} \
      --set image.tag=${{ steps.meta.outputs.tags }} \
      --set config.githubBranch=${{ env.GITHUB_BRANCH }} \
      --set config.prefectApiUrl=${{ env.PREFECT_API_URL }} \
      --set secrets.githubToken=${{ secrets.GITHUB_TOKEN }} \
      --set secrets.prefectGithubCredentials=${{ secrets.PREFECT_GITHUB_CREDENTIALS }} \
      --set secrets.postgresUrl=${{ secrets.POSTGRES_URL }}
```

### Manual Deployment

1. **Install the chart:**
   ```bash
   cd charts/deploy-prefect
   helm install deploy-prefect . \
     --set image.repository=ghcr.io/your-org/your-image \
     --set image.tag=v1.0.0 \
     --set config.githubBranch=main \
     --set config.prefectApiUrl=https://prefect.example.com/api \
     --set env.GITHUB_TOKEN=your-token \
     --set env.PREFECT_GITHUB_CREDENTIALS=your-credentials \
     --set env.POSTGRES_URL=your-postgres-url
   ```

2. **Upgrade existing deployment:**
   ```bash
   helm upgrade deploy-prefect . \
     --set image.tag=v1.1.0
   ```

3. **Uninstall:**
   ```bash
   helm uninstall deploy-prefect
   ```

## Configuration

### Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Docker image repository | `ghcr.io/your-org/mc-prefect-loaders` |
| `image.tag` | Docker image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `Always` |
| `config.githubBranch` | GitHub branch name | `main` |
| `config.prefectApiUrl` | Prefect API URL | `https://prefect.example.com/api` |
| `env.GITHUB_TOKEN` | GitHub token | `""` |
| `env.PREFECT_GITHUB_CREDENTIALS` | Prefect GitHub credentials | `""` |
| `env.POSTGRES_URL` | PostgreSQL URL | `""` |
| `env.PYTHONUNBUFFERED` | Python unbuffered output | `"1"` |
| `env.PYTHONPATH` | Python path | `"/app"` |
| `env.LOG_LEVEL` | Logging level | `"INFO"` |
| `env.ENVIRONMENT` | Environment name | `"production"` |
| `env.DEBUG` | Debug mode | `"false"` |
| `resources.requests.memory` | Memory request | `128Mi` |
| `resources.requests.cpu` | CPU request | `100m` |
| `resources.limits.memory` | Memory limit | `256Mi` |
| `resources.limits.cpu` | CPU limit | `200m` |
| `job.parallelism` | Job parallelism | `1` |
| `job.completions` | Job completions | `1` |
| `job.activeDeadlineSeconds` | Job timeout | `1800` (30 min) |
| `job.ttlSecondsAfterFinished` | Cleanup delay | `300` (5 min) |
| `job.backoffLimit` | Retry limit | `3` |

### Override Values

Create a custom `values.yaml` file:

```yaml
image:
  repository: ghcr.io/your-org/custom-image
  tag: v2.0.0

config:
  githubBranch: develop
  prefectApiUrl: https://staging.prefect.example.com/api

resources:
  requests:
    memory: "256Mi"
    cpu: "200m"
```

Then deploy with:
```bash
helm install deploy-code . -f custom-values.yaml
```

### Setting Environment Variables via Command Line

You can also override environment variables directly:

```bash
helm install deploy-code . \
  --set env.LOG_LEVEL=DEBUG \
  --set env.ENVIRONMENT=staging \
  --set env.DEBUG=true \
  --set env.CUSTOM_VAR=my_value
```

### Setting Environment Variables in GitHub Actions

```yaml
- name: Deploy to Digital Ocean cluster
  run: |
    cd charts/deploy-prefect
    helm upgrade --install deploy-prefect . \
      --set image.repository=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }} \
      --set image.tag=${{ steps.meta.outputs.tags }} \
      --set config.githubBranch=${{ env.GITHUB_BRANCH }} \
      --set config.prefectApiUrl=${{ env.PREFECT_API_URL }} \
      --set env.GITHUB_TOKEN=${{ secrets.GITHUB_TOKEN }} \
      --set env.PREFECT_GITHUB_CREDENTIALS=${{ secrets.PREFECT_GITHUB_CREDENTIALS }} \
      --set env.POSTGRES_URL=${{ secrets.POSTGRES_URL }} \
      --set env.ENVIRONMENT=${{ env.ENVIRONMENT }} \
      --set env.LOG_LEVEL=${{ env.LOG_LEVEL }} \
      --set env.CUSTOM_VAR=${{ env.CUSTOM_VAR }}
```

## What the Job Does

1. **Sets Variables** - Runs `data/variables_base.py` to configure Prefect API URL and GitHub branch
2. **Sets Secrets** - Runs `data/secrets_base.py` to configure GitHub token, Prefect credentials, and PostgreSQL URL
3. **Deploys Code** - Finds and runs all `*_deploy.py` files recursively in the `src` directory

## Benefits of Helm

- **Clean separation** of configuration from manifests
- **Easy parameterization** without complex sed commands
- **Version control** for your deployments
- **Rollback capability** with `helm rollback`
- **Template inheritance** and reusable components
- **Standard Kubernetes tooling** that teams are familiar with
