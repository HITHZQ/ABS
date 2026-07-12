# Ziqi Hao - PhD Interview Presentation Script\n\n**Title:** Toward Simulation-Ready World Models for Embodied Robot Learning\n\n## Slide 1: Research Experience and Future Research Vision\n\nGood morning, Professor Wang. Thank you very much for meeting with me.

Today, I will present my previous research experience and my future research vision. My presentation is centered around one question: how can robots truly understand and interact with the physical world?

My background is in robotics, reinforcement learning, and control. Through my previous projects, I gradually realized that robot learning depends not only on better control policies, but also on better representations and simulations of the physical world. This is why I am very interested in simulation-ready world models for embodied robot learning.\n\n## Slide 2: About Me: Robotics, Learning, and Control\n\nI am currently an undergraduate student in Automation at Harbin Institute of Technology, expecting to receive my bachelor's degree in 2026.

My research experience is mainly in robotics, learning-based control, and simulation. I have worked on humanoid whole-body motion retargeting, quadruped locomotion with reinforcement learning, SE(3) control for quadrotors, and several projects related to trajectory optimization and robot planning.

My long-term goal is to build embodied AI systems that can perceive, understand, simulate, and interact with the physical world. I see my previous robotics experience as a foundation for this broader research direction.\n\n## Slide 3: Motivation: Why World Models Matter for Robot Learning\n\nMy motivation comes from a common problem in robot learning.

Many robot policies are trained in simulation before being deployed in the real world. However, simulation environments are often manually designed, limited in diversity, and not always physically accurate. This creates a gap between what the robot learns in simulation and what it faces in the real world.

I became interested in the idea that better physical world models could improve this pipeline. If we can reconstruct dynamic environments from real observations, represent geometry, motion, contact, and physical properties, and turn these representations into simulation-ready environments, then robot learning may become more scalable and generalizable.\n\n## Slide 4: Humanoid Whole-Body Motion Retargeting and RL Tracking\n\nThe first project I would like to discuss is humanoid whole-body motion retargeting and reinforcement-learning-based tracking.

The goal is to transfer expressive human or reference motions to humanoid robots. This is challenging because human motions are not directly executable by humanoid robots. The robot has different morphology, different joint limits, limited torque capability, and needs to satisfy balance and contact constraints.

We used a two-stage pipeline. First, reference motions are retargeted to the humanoid body while considering kinematic feasibility. Then, an RL-based tracking policy is trained in simulation to robustly execute the retargeted motions.

My main contribution was not only running an existing RL pipeline. I worked on the motion retargeting process, simulation validation, and RL-based tracking policy training. Through this project, I found that many failures were not simply caused by weak policies, but by deeper physical constraints such as contact, balance, joint limits, and dynamics mismatch.\n\n## Slide 5: Physical Feasibility and Simulation Matter\n\nThis project taught me an important lesson: motion imitation is not only a policy learning problem.

Even if a reference motion looks natural for a human, it may be physically infeasible for a humanoid robot. The final behavior depends on contact dynamics, body geometry, joint constraints, actuation limits, and the physical environment.

Through this project, I realized that policy learning is only one part of the problem. The quality of simulation and the quality of the physical representation can strongly affect what a robot can learn and how well the learned behavior can transfer.

This motivated me to think beyond policy learning itself and become interested in physical world models that can support embodied robot learning.\n\n## Slide 6: Quadruped Locomotion with Reinforcement Learning\n\nMy second project is quadruped locomotion with reinforcement learning.

The goal was to train robust locomotion policies for a quadruped robot under different dynamics and terrain conditions. We used end-to-end reinforcement learning in simulation, with reward terms related to velocity tracking, stability, smoothness, and energy efficiency.

To improve robustness, we also used domain randomization and evaluated policies under different simulation settings. My contributions included implementing and tuning the training pipeline, designing reward terms, and evaluating robustness and failure modes.

This project further strengthened my understanding that simulation quality is crucial for robot learning. If important physical variations or contact conditions are missing in simulation, the learned policy may still fail during deployment, even if the RL algorithm itself is strong.\n\n## Slide 7: Common Bottleneck: Simulation Quality and Physical Realism\n\nLooking back at these projects, I found a common bottleneck.

Humanoid motion learning depends on physical simulation. Quadruped locomotion depends on robust simulated environments. For manipulation and interaction tasks, the robot further needs to understand geometry, contact, object dynamics, and material properties.

So the common bottleneck is not only how to learn a policy, but also how to build realistic and physically meaningful environments in which policies can be learned and evaluated.

This naturally leads to my current research interest: can we build simulation-ready world models from real observations?\n\n## Slide 8: Why Prof. Wang's Research Directly Connects to My Problem\n\nThis is where I see a strong connection with your research.

From my robotics projects, I found that robot learning needs simulation environments that capture geometry, motion, contact, physical properties, and dynamic scene evolution. However, current simulations are often manually built and difficult to scale to diverse real-world scenarios.

This is why I find your research highly relevant. Works such as PhysConvex, ArtMesh, and InNeRF provide important building blocks for reconstructing dynamic, structured, and physically meaningful worlds from observations.

For me, the key question is: can these physical world representations become environments where embodied agents can learn, plan, and interact?\n\n## Slide 9: Future Vision: Simulation-Ready Physical World Models\n\nFor my future PhD research, I hope to work toward simulation-ready physical world models for embodied robot learning.

A possible pipeline starts from videos or multimodal observations. From these observations, we reconstruct a 3D or 4D representation of the scene. Then, we enrich this representation with physical properties, motion, contact, and dynamics, so that it can become a simulation-ready environment.

Finally, robots can use these reconstructed environments for policy learning, planning, and interaction.

I believe this direction naturally connects computer vision, graphics, physical simulation, and robotics.\n\n## Slide 10: Research Questions\n\nMore specifically, I am interested in several research questions.

First, how can robots reconstruct dynamic physical worlds from sparse visual or multimodal observations?

Second, how can reconstructed 3D or 4D scenes become simulation-ready, rather than only visually realistic?

Third, how can these physical world models improve policy learning and generalization?

Fourth, how can embodied agents interact safely with dynamic environments?

Finally, I am also interested in interpretability and trustworthiness. If a robot uses a world model to make decisions, we need to understand what the model represents and where it may fail.\n\n## Slide 11: Why My Background Fits This Direction\n\nI believe my background fits this direction because it provides a robotics-centered perspective.

I started from robotics and control, then worked on learning-based policies in simulation. These projects exposed me to the sim-to-real problem and made me realize the importance of physical world models.

From humanoid and quadruped projects, I gained experience with robot motion, simulation, reinforcement learning, and sim-to-real challenges. From SE(3) control and trajectory optimization, I developed a foundation in dynamics and control.

Therefore, I am not moving away from robotics. Instead, I hope to extend my robotics background toward 3D reconstruction, physics-informed simulation, and simulation-ready world modeling.\n\n## Slide 12: Why Prof. Wang's Group\n\nI am very interested in your group because your research bridges computer vision, graphics, and artificial intelligence, with a strong focus on physical scene understanding and simulation.

Your recent work on physics-informed reconstruction, dynamic world modeling, and interpretable neural representations is closely aligned with the direction I hope to pursue.

I believe there is a natural complementarity. Your research provides powerful tools for reconstructing and simulating dynamic physical worlds. My background in robotics, reinforcement learning, and control motivates me to explore how such representations can support embodied robot learning and interaction.

This is why I am very excited about the possibility of joining your group.\n\n## Slide 13: Long-Term Vision\n\nTo conclude, my previous research started from robot motion learning and control. Through humanoid and quadruped projects, I became increasingly interested in a broader question: how can robots learn from realistic physical world models instead of only manually designed simulations?

For my PhD, I hope to connect robotics, physical simulation, and 3D world modeling. I hope to contribute to the next generation of embodied AI by connecting physical world modeling with robot learning.

Thank you very much. I would be happy to discuss any questions.\n\n