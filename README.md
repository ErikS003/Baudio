# Beehive Queen Presence Prediction

This project uses **machine learning and audio analysis** to predict whether the queen of a beehive is present based on **sound recordings**. 
Bees produce distinctive sounds depending on the hive’s state. The presence (or absence) of the queen can be inferred by analyzing these sounds.  

This project leverages a machine learning (ML) model trained on existing beehive audio datasets to make binary predictions:
- **1 → Queen present**  
- **0 → Queen absent**  

The long-term goal is to integrate this model with **real-time audio streams from our own hive** and automatically store predictions using **AWS infrastructure**.

---

##  Current Progress
ML model trained on publicly available beehive sound recordings.  
Model outputs binary predictions (queen present/absent).  
Basic local inference pipeline working with audio datasets.  

---

## Work in Progress
- Collect in-house beehive recordings for **testing and validation** on real-world data.  
- Automate data pipeline:
  - Capture audio from a microphone installed on the hive.  
  - Preprocess audio and send it through the trained ML model.  
  - Store predictions (and possibly raw audio) in **AWS**.  
- Deploy the model for real-time or batch predictions in the cloud or locally on a rpi.  
- Build monitoring and visualization tools for hive health tracking.  
