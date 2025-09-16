# Chaos Tests

These tests apply Chaos Mesh experiments in the staging namespace to validate recovery and autoscaling behaviors.

Notes:
- Requires a connected Kubernetes context with access to the staging cluster.
- Ensure Chaos Mesh is installed via k8s/staging/chaos-mesh kustomization.
- Set env NS=agentforge-staging if your namespace differs.
