# 🤖 RL_Projects – Reinforcement Learning Projects and Experiments

This repository documents my hands-on exploration of reinforcement learning (RL) in the context of robotics, simulation, and control systems. It follows a 12-week structured learning plan, combining theoretical study with practical implementation — targeting mastery in tools like Isaac Gym and deep reinforcement learning algorithms.

---

## 📚 Table of Contents

| Week | Project | Description |
|------|---------|-------------|
| 1 | [CartPole DQN](./week01_dqn_cartpole/) | Implementation of a 2D cart balancing a pole (1 DOF) using Deep Q-Learning. Based on the [PyTorch RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). |
| 2 | [MuJoCo PPO/SAC](./week02_sac_halfcheetah/) | PPO and SAC applied to continuous control tasks like `HalfCheetah` and `Walker2d`. |
| 3 | [Custom RL Baselines](./week03_rl_baselines/) | Reimplementation of PPO and SAC from scratch with GAE and modular PyTorch code. |
| 4 | [Robotic Arm Grasping (Isaac Gym)](./week04_franka_grasping/) | Use PPO/SAC to teach a simulated Franka Panda arm to grasp a cube. |
| 5 | [Cloud Training Deployment](./week05_cloud_training/) | Train agents remotely using AWS EC2 or Lambda Labs + Ray RLlib. |
| 6 | [Humanoid Locomotion](./week06_humanoid_locomotion/) | Teach a humanoid robot to walk using Isaac Gym + PPO + curriculum learning. |
| 7 | [Dexterous Manipulation](./week07_dexterous_hand/) | Use a ShadowHand model to manipulate objects with fine motor control. |
| 8 | [RL Competition Entry](./competition_aicrowd/) | Train and submit an agent to an open RL competition (e.g. AIcrowd). |
| 9–12 | [Final Project & Portfolio Polish](./final_project_optimus_sim/) | Showcase project inspired by Tesla Optimus: full-body locomotion + manipulation.

---

## 🧠 Topics Covered

- Deep RL: DQN, PPO, SAC, GAE
- Custom reward shaping, curriculum learning
- Isaac Gym (NVIDIA) for high-performance robotic sim
- Visual input + partial observability
- Cloud deployment & RLlib
- Portfolio & interview prep

---

## 📸 Demo Media

> Each project folder includes sample output videos, training logs, and architecture diagrams. A short demo reel will be available after Week 12.

---

## 📝 Blog Posts

Coming soon:
- **How I Built a Grasping Robot in Isaac Gym (Week 4)**
- **What Tesla Optimus Taught Me About RL (Final Project)**

---

## 🧰 Tools & Environment

- Python 3.8+
- PyTorch
- Isaac Gym (Preview 4)
- MuJoCo / Gymnasium
- Ubuntu 22.04 (dual boot)
- CUDA 11.6
- wandb / Ray / RLlib

---

## 🙋 About Me

I'm Jonas Petersen, aspiring robotics & AI engineer with a background in mechanical engineering, embedded systems, and a passion for embodied intelligence and startups.

📬 Reach out:  
- [LinkedIn](https://linkedin.com/in/7jep7)  
- [je77petersen@gmail.com](mailto:je77petersen@gmail.com)

---

## 🚀 Let's teach robots to learn.
