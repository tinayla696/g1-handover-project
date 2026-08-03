---
name: "🚀 Multi-Instance Verification Report"
about: Share training progress, visual checks, and reproduction status across different AWS instances.
title: "[Verification] g6.12xlarge - Portability & Jitter Test"
labels: ["verification", "infrastructure"]
assignees: ""
---

## 💻 Environment Metadata
- **AWS Instance Type:** (e.g., g6.12xlarge / g5.24xlarge)
- **Availability Zone:** (e.g., us-east-1a)
- **Base AMI / OS:** Ubuntu 24.04 LTS (Noble Numbat)
- **Isaac Lab Commit Hash:** `6a7acb0320a0bdc15b13e44e83b575e00797faf4`

---

## 👁️ Visual Check Status (Before RL)
- [ ] **Docker Build Success:** Container initialized without dependency loops.
- [ ] **S3 Pull Success:** Previous checkpoints fetched correctly.
- [ ] **WebRTC Connection:** Stream connected successfully via Public IP.
- [ ] **Asset Alignment Check:** 
  - *Robot Initial Pose:* (OK / Jittery / Erroneous)
  - *Table & Novelty Object (Bottle) Collision:* (No collision / Sinking / Floating)

*Notes on Visuals:* (Add screenshots or description of the initial scene here)

---

## 📊 Reinforcement Learning Progress
- **Target Iterations:** 1000
- **Actual Completed Iterations:** 
- **S3 Model Upload Status:** (Synced / Failed)

### 📈 Training Metrics & Bottlenecks
- **Mean Reward Trend:** (Flat / Improving / Dropping)
- **Termination Causes:** (Time-out reached / High rate of `robot_fall`)
- **Action Scale / Height Threshold applied:** (e.g., action_scale=0.05, root_height=0.3)

---

## 📝 Key Takeaways & Next Actions for Other Instances/AIs
- (Example: "g6.12xlarge runs stable but training speed drops by X% compared to g7e. Need to adjust batch size.")