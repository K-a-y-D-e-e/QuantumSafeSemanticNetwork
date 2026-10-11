\# RL Handoff — Semantic Communication



\## 1. Current status



The semantic compression environment, rule-based orchestration, and PPO evaluation pipeline are implemented. The current test command passes:



```bash

python -m pytest test\_semantic\_rl.py -q

```



Latest run: \*\*15 passed\*\*.



\## 2. Relevant files



\- `semantic/encoder.py`, `semantic/decoder.py`, `semantic/model\_io.py` — semantic model implementation and loading.

\- `rl/environment.py` — `SemanticCompressionEnv` and the separate `NetworkSchedulingEnv`.

\- `orchestration/agent.py` — rule-based objective selection and action biasing.

\- `train\_semantic\_rl.py` — PPO training entry point.

\- `evaluate\_semantic\_rl.py` — baseline and PPO evaluation.

\- `semantic\_encoder.pth`, `semantic\_decoder.pth` — trained semantic model weights.

\- `rl/checkpoints/ppo\_semantic\_adaptive.zip` — existing PPO checkpoint.



\## 3. Semantic PPO interface



Observation: 7 normalized values in this order:



1\. Task criticality

2\. Available bandwidth

3\. Network load

4\. Latency

5\. Packet loss

6\. Deadline

7\. Semantic quality



Actions:



\- `0` — 25% retention

\- `1` — 50% retention

\- `2` — 75% retention

\- `3` — 100% retention



The environment supports analytic mode and reconstruction-based reward mode. Reconstruction mode uses the semantic encoder, compressor, and decoder.



\## 4. Orchestration



The orchestration agent selects a rule-based objective and can bias the PPO compression action. It is not a learned agent.



The `BALANCED` objective's quality weight is currently `2.5`, changed from `1.0` in the current working changes. Confirm this configuration before further training.



\## 5. Latency proxy



The semantic environment's latency estimate is a heuristic proxy, not measured end-to-end network latency.



The configurable `latency\_compression\_weight` defaults to `0.35`. A comparison with `0.15` was run on 256 held-out samples. The lower coefficient reduced the estimated latency, but did not change reconstruction MSE or the existing PPO policy's actions. The candidate coefficient has not been validated against NS-3 measurements.



\## 6. Existing evaluation caveat



In the latest 256-sample comparison, raw PPO matched the fixed 25% baseline metrics. Inspect the selected-action distribution before assuming PPO has learned useful adaptive behaviour.



\## 7. Suggested next steps



1\. Confirm model weights and the PPO checkpoint load successfully.

2\. Inspect PPO's selected-action distribution.

3\. Validate the reward configuration and compare PPO against fixed-retention baselines.

4\. Continue DQN network-scheduling work independently where possible.

5\. Integrate both agents with NS-3 and evaluate actual latency, jitter, packet delivery, and throughput.



Do not claim NS-3 performance improvements based solely on the current latency proxy.

