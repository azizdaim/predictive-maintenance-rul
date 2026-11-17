---
title: Turbofan Engine Remaining Life
emoji: 🛫
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: Predict how many flight cycles a turbofan engine has left
---

A dashboard that predicts the remaining useful life of turbofan engines from
their sensor history, using NASA's C-MAPSS dataset. A LightGBM model is served
by a FastAPI service, and both run in this Space's Docker container.

Source code, tests, and results: https://github.com/azizdaim/predictive-maintenance-rul
