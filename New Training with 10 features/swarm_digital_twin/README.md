# Swarm Digital Twin Extension

This folder contains the PSO-SVM swarm-intelligence model and supporting outputs for the top-10 PD speech feature website.

## PSO-SVM

- Particle Swarm Optimization searched SVM `C` and `gamma` on the same 10 mutual-information-selected features.
- Best `C`: 45.2293
- Best `gamma`: 0.0100072
- Search mean F1: 0.9034
- 10-fold mean accuracy: 0.8439
- 10-fold mean recall: 0.9628
- 10-fold mean F1: 0.9021
- 10-fold mean ROC-AUC: 0.8576

## Digital Twin

The website uses prediction responses to store a local baseline voice profile and compare later predictions against it. No patient data is stored on the backend.
