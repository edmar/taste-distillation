# Report

## Shortlist Classifier


### No rubric

uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1-mini --no-rubric

📂 Dataset: data/reader_shortlist/test/dspy_examples.json
Evaluating on full test set: 134 examples
🚫 Skipping rubric (--no-rubric flag)
🤖 Using LLM: openai/gpt-4.1-mini
📊 Running baseline evaluation with untrained model
Average Metric: 76.00 / 134 (56.7%): 100%|##########| 134/134 [00:23<00:00,  5.67it/s]
2025/07/05 18:46:27 INFO dspy.evaluate.evaluate: Average Metric: 76 / 134 (56.7%)
🎯 Accuracy: 56.720
✅ Log saved to: saved/logs/dspy_shortlist_eval_035.log
✅ Evaluation completed! Accuracy: 56.720


uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1 --no-rubric

📂 Dataset: data/reader_shortlist/test/dspy_examples.json
Evaluating on full test set: 134 examples
🚫 Skipping rubric (--no-rubric flag)
🤖 Using LLM: openai/gpt-4.1
📊 Running baseline evaluation with untrained model
Average Metric: 70.00 / 134 (52.2%): 100%|##########| 134/134 [00:28<00:00,  4.62it/s]
2025/07/05 18:43:52 INFO dspy.evaluate.evaluate: Average Metric: 70 / 134 (52.2%)
🎯 Accuracy: 52.240
✅ Log saved to: saved/logs/dspy_shortlist_eval_033.log
✅ Evaluation completed! Accuracy: 52.240


### Test with Rubric

uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1-mini

📂 Dataset: data/reader_shortlist/test/dspy_examples.json
Evaluating on full test set: 134 examples
✅ Loaded taste rubric
🤖 Using LLM: openai/gpt-4.1-mini
📊 Running baseline evaluation with untrained model
Average Metric: 90.00 / 134 (67.2%): 100%|##########| 134/134 [00:32<00:00,  4.07it/s]
2025/07/05 18:35:15 INFO dspy.evaluate.evaluate: Average Metric: 90 / 134 (67.2%)
🎯 Accuracy: 67.160
✅ Log saved to: saved/logs/dspy_shortlist_eval_026.log
✅ Evaluation completed! Accuracy: 67.160

uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1

📂 Dataset: data/reader_shortlist/test/dspy_examples.json
Evaluating on full test set: 134 examples
✅ Loaded taste rubric
🤖 Using LLM: openai/gpt-4.1
📊 Running baseline evaluation with untrained model
Average Metric: 94.00 / 134 (70.1%): 100%|##########| 134/134 [00:25<00:00,  5.32it/s]
2025/07/05 18:37:15 INFO dspy.evaluate.evaluate: Average Metric: 94 / 134 (70.1%)
🎯 Accuracy: 70.150
✅ Log saved to: saved/logs/dspy_shortlist_eval_027.log
✅ Evaluation completed! Accuracy: 70.150

uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/o3

📂 Dataset: data/reader_shortlist/test/dspy_examples.json
Evaluating on full test set: 134 examples
✅ Loaded taste rubric
🤖 Using LLM: openai/o3
📊 Running baseline evaluation with untrained model
Average Metric: 93.00 / 134 (69.4%): 100%|##########| 134/134 [00:56<00:00,  2.36it/s]
2025/07/05 18:40:12 INFO dspy.evaluate.evaluate: Average Metric: 93 / 134 (69.4%)
🎯 Accuracy: 69.400
✅ Log saved to: saved/logs/dspy_shortlist_eval_030.log
✅ Evaluation completed! Accuracy: 69.400


