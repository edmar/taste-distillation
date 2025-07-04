# Taste Distillation Model Evaluation Report


## Baseline 
uv run python src/dspy_pairwise/scripts/evaluate.py

==================================================
DSPy PAIRWISE CLASSIFIER EVALUATION
==================================================
📂 Loading dataset from: data/processed/reader_pairwise/test/pairwise_classifier.json
Evaluating on full test set: 300 examples
✅ Loaded taste rubric
📊 Running baseline evaluation with untrained model
Average Metric: 184.00 / 300 (61.3%): 100%|█████████████████████████████████████████████████████████████████████████| 300/300 [01:00<00:00,  4.98it/s]
2025/07/03 20:53:10 INFO dspy.evaluate.evaluate: Average Metric: 184 / 300 (61.3%)

🎯 Accuracy: 61.330

✅ Evaluation completed! Accuracy: 61.330



uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/processed/hn_pairwise/val/dspy_examples.json

==================================================
DSPy PAIRWISE CLASSIFIER EVALUATION
==================================================
📂 Loading dataset from: data/processed/hn_pairwise/val/dspy_examples.json
Evaluating on full test set: 225 examples
✅ Loaded taste rubric
📊 Running baseline evaluation with untrained model
Average Metric: 147.00 / 225 (65.3%): 100%|█████████████████████████████████████████████████████████████████████████| 225/225 [00:45<00:00,  4.98it/s]
2025/07/03 20:54:24 INFO dspy.evaluate.evaluate: Average Metric: 147 / 225 (65.3%)

🎯 Accuracy: 65.330

✅ Evaluation completed! Accuracy: 65.330


## Training


uv run python src/dspy_pairwise/scripts/train.py --dataset data/processed/hn_pairwise/train/dspy_examples.json

2025/07/03 21:03:08 INFO dspy.teleprompt.mipro_optimizer_v2: ===== Trial 13 / 13 - Full Evaluation =====
2025/07/03 21:03:08 INFO dspy.teleprompt.mipro_optimizer_v2: Doing full eval on next top averaging program (Avg Score: 87.14500000000001) from minibatch trials...
Average Metric: 82.00 / 100 (82.0%): 100%|██████████████████████████████████████████████████████████████████████████| 100/100 [00:25<00:00,  3.89it/s]
2025/07/03 21:03:34 INFO dspy.evaluate.evaluate: Average Metric: 82 / 100 (82.0%)
2025/07/03 21:03:34 INFO dspy.teleprompt.mipro_optimizer_v2: Full eval scores so far: [86.0, 87.0, 82.0]
2025/07/03 21:03:34 INFO dspy.teleprompt.mipro_optimizer_v2: Best full score so far: 87.0
2025/07/03 21:03:34 INFO dspy.teleprompt.mipro_optimizer_v2: =======================
2025/07/03 21:03:34 INFO dspy.teleprompt.mipro_optimizer_v2:

2025/07/03 21:03:34 INFO dspy.teleprompt.mipro_optimizer_v2: Returning best identified program with score 87.0!
✅ Optimization completed successfully!

💾 Saving model...
✅ Model saved to: saved/models/dspy_pairwise_004.json
✅ Metadata saved to: saved/models/dspy_pairwise_004_metadata.json

✅ Training completed! Model saved to: saved/models/dspy_pairwise_004.json


## Model Evaluation

uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_004.json
==================================================
DSPy PAIRWISE CLASSIFIER EVALUATION
==================================================
📂 Loading dataset from: data/processed/reader_pairwise/test/pairwise_classifier.json
Evaluating on full test set: 300 examples
✅ Loaded taste rubric
✅ Loaded trained model from: saved/models/dspy_pairwise_004.json
Average Metric: 169.00 / 300 (56.3%): 100%|█████████████████████████████████████████████████████████████████████████| 300/300 [01:27<00:00,  3.44it/s]
2025/07/03 21:09:41 INFO dspy.evaluate.evaluate: Average Metric: 169 / 300 (56.3%)

🎯 Accuracy: 56.330

✅ Evaluation completed! Accuracy: 56.330



uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_004.json --dataset data/processed/hn_pairwise/val/dspy_examples.json
==================================================
DSPy PAIRWISE CLASSIFIER EVALUATION
==================================================
📂 Loading dataset from: data/processed/hn_pairwise/val/dspy_examples.json
Evaluating on full test set: 225 examples
✅ Loaded taste rubric
✅ Loaded trained model from: saved/models/dspy_pairwise_004.json
Average Metric: 173.00 / 225 (76.9%): 100%|█████████████████████████████████████████████████████████████████████████| 225/225 [00:59<00:00,  3.81it/s]
2025/07/03 21:12:43 INFO dspy.evaluate.evaluate: Average Metric: 173 / 225 (76.9%)

🎯 Accuracy: 76.890

✅ Evaluation completed! Accuracy: 76.890


 uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_004.json --dataset data/processed/hn_pairwise/test/dspy_examples.json
==================================================
DSPy PAIRWISE CLASSIFIER EVALUATION
==================================================
📂 Loading dataset from: data/processed/hn_pairwise/test/dspy_examples.json
Evaluating on full test set: 225 examples
✅ Loaded taste rubric
✅ Loaded trained model from: saved/models/dspy_pairwise_004.json
Average Metric: 171.00 / 225 (76.0%): 100%|█████████████████████████████████████████████████████████████████████████| 225/225 [01:02<00:00,  3.60it/s]
2025/07/03 21:16:26 INFO dspy.evaluate.evaluate: Average Metric: 171 / 225 (76.0%)

🎯 Accuracy: 76.000

✅ Evaluation completed! Accuracy: 76.000


