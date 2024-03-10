# concept-learning

## Data Preparation

### Concepts

- CUB:
    - [LaBo all concepts](https://github.com/YueYANG1996/LaBo/tree/main/datasets/CUB/concepts)
    - [LCDA](https://github.com/wangyu-ustc/LM4CV/blob/main/data/CUB_200_2011/cub_attributes.txt)

To encode concepts, for example:
```bash
python prepare_concepts.py \
--dataset CUB_200_2011 \
--output_dir concepts/CUB_200_2011/LCDA \
--concept_path concepts/CUB_200_2011/LCDA/concepts.txt
```