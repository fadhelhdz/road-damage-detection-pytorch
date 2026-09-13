# Problem Statement — Road Damage Detection & Condition Monitoring
- Task type: Detection problem - Each picture contains unknown number of cracks/potholes with unknown location, and each needs class and box defined. 
- Inputs / Outputs: The inputs are image, the outputs are bounding box (showing location) and class (classify the type of the cracks or potholes, the confidence probability of the detected object is really cracks)
- Classes: Longitudinal Crack (D00), Transverse Crack (D10), Alligator Crack (D20), Pothole (D40)
- Primary metric: mAP@50:95, which summarizes object detection and localization performance across all classes, confidence thresholds, and multiple IoU thresholds.
- Secondary metrics: AP50, which measures detection performance using a relatively lenient IoU threshold of 0.50, making it useful for assessing whether the detector can correctly identify and roughly localize objects.
- Business scenario I'm optimizing for: budgeting
- How that scenario sets my threshold/recall-vs-precision tradeoff: false positive increase the budget, because the detector overestimate the potholes, so budget increase, precision matters more.
## Measurable success criteria (targets, not yet measured)
- MVP target: AP50 ≥ 0.70 on the validation set — TARGET, not yet measured. cracks have challenging appearance, thin/irregular shapes, variable lighting, and background confusion. An AP50 of 0.70 would indicate that the model can reliably detect and roughly localize cracks while leaving room for false positives and missed detections.
- Latency target on RTX 3060 6GB: ≤ 30 ms/image for model inference
- Reproducibility criterion: With the same dataset split, model configuration, software environment, and random seed, repeated training runs should produce AP50 within ±2 percentage points of the reference result.
Report the seed, dataset/version, model configuration, and evaluation procedure so the result can be independently reproduced.