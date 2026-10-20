#!/usr/bin/env bash
# Deploy the CardioRisk API and its monitoring stack (Prometheus + Grafana) to the
# current kubectl context (Docker Desktop, minikube, AKS, ...).
#
# usage: bash scripts/k8s_deploy.sh
#        IMAGE=cardiorisk-api:1.0.0 bash scripts/k8s_deploy.sh   # locally loaded image (minikube)
set -euo pipefail
cd "$(dirname "$0")/.."

PUBLISHED_IMAGE="ghcr.io/akshaya-panda/mlops-assignment:latest"

if ! command -v kubectl >/dev/null 2>&1; then
  if command -v minikube >/dev/null 2>&1; then
    kubectl() { minikube kubectl -- "$@"; }
  else
    echo "kubectl not found"; exit 1
  fi
fi

echo "==> kubectl context: $(kubectl config current-context)"

echo "==> API (namespace cardiorisk)"
if [[ -n "${IMAGE:-}" ]]; then
  echo "    using image ${IMAGE} (IfNotPresent)"
  kubectl kustomize k8s/ \
    | sed -e "s#${PUBLISHED_IMAGE}#${IMAGE}#" -e "s#imagePullPolicy: Always#imagePullPolicy: IfNotPresent#" \
    | kubectl apply -f -
else
  kubectl apply -k k8s/
fi
kubectl -n cardiorisk rollout status deployment/cardiorisk-api --timeout=300s

echo "==> monitoring (namespace monitoring)"
kubectl apply -f monitoring/k8s/namespace.yaml
configmap() {
  kubectl -n monitoring create configmap "$1" "${@:2}" --dry-run=client -o yaml | kubectl apply -f -
}
configmap prometheus-config --from-file=monitoring/prometheus/
configmap grafana-datasources --from-file=monitoring/grafana/provisioning/datasources/
configmap grafana-dashboard-provider --from-file=monitoring/grafana/provisioning/dashboards/
configmap grafana-dashboards --from-file=monitoring/grafana/dashboards/

if ! kubectl -n monitoring get secret grafana-admin >/dev/null 2>&1; then
  # Generated once and stored only in the cluster; never printed or committed.
  kubectl -n monitoring create secret generic grafana-admin \
    --from-literal=password="$(openssl rand -hex 16)" >/dev/null
  echo "    created secret monitoring/grafana-admin"
fi

# Restart each component only when its own configuration changed.
apply_with_hash() {  # $1 = manifest, remaining args = config files it depends on
  local hash
  hash="$(cat "${@:2}" | sha256sum | cut -c1-12)"
  sed "s#cardiorisk/config-hash: \"initial\"#cardiorisk/config-hash: \"${hash}\"#" "$1" \
    | kubectl apply -f -
}
apply_with_hash monitoring/k8s/prometheus.yaml monitoring/prometheus/*
apply_with_hash monitoring/k8s/grafana.yaml monitoring/grafana/provisioning/*/* \
  monitoring/grafana/dashboards/*
kubectl -n monitoring rollout status deployment/prometheus --timeout=300s
kubectl -n monitoring rollout status deployment/grafana --timeout=300s

echo
kubectl -n cardiorisk get deployments,pods,services -o wide
echo
kubectl -n monitoring get deployments,pods,services
cat <<'EOF'

Endpoints (Docker Desktop publishes LoadBalancer services on localhost;
on minikube run `minikube tunnel` in another terminal first):
  API         http://127.0.0.1:8000/docs
  Prometheus  http://127.0.0.1:9090/targets
  Grafana     http://127.0.0.1:3001   (anonymous read-only)
Grafana admin password:
  kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d
EOF
