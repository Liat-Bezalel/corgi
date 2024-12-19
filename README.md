# Teaching Models to Improve on Tape
Codebase for AAAI paper ["Teaching Models to Improve on Tape"](https://arxiv.org/abs/2411.01483)

## Training

To train the model, you can use the following command:

```bash
python trainer_feedback.py --task <TASK> --retry_limit <RETRY_LIMIT> --model_name <MODEL_NAME> --batch_size <BATCH_SIZE> --model_type <MODEL_TYPE> --use_peft
```

### Parameters
\<TASK\>: The task to train on. It can be one of the following:
- meta_learning: Train jointly on the source tasks.
- Specific task from the list: num_planning, paraphrase, rationale_generation, sentiment, story_generation.

\<RETRY_LIMIT\>: The number of retry attempts provided to the model. For example, 3 means the model will have a total of 4 attempts (1 initial + 3 retries).

\<MODEL_NAME\>: The pre-trained model to use. Example: meta-llama/Meta-Llama-3-8B-Instruct.

\<BATCH_SIZE\>: The batch size to use during training (e.g., 32).

\<MODEL_TYPE\>: Currently supported llama_2 or llama_3.


## Evaluation

To evaluate the model, use the following command:

```bash
python validate_with_critique.py --model_prefix_path <MODEL_NAME> --task <TASK> --retry_limit <RETRY_LIMIT> --split <SPLIT> --batch_size <BATCH_SIZE> --model_type <MODEL_TYPE> --max_len <MAX_LEN>
```
