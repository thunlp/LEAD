## Dataset

**You can find the LEAD dataset [here](https://huggingface.co/datasets/JamesChengGao/LEAD)**. 

0, 30, 65, and 100 in the file names represent the proportion of query-key pairs that belong to the same case. Please refer to the ablation section of our paper for more details.

The simplified version of LeCaRD queries is in `LeCaRD/data/query/query_simplify.json` and the simplified version of CAIL2022 queries is in `CAIL2022/stage2/query_stage2_valid_onlystage2_40_simplified.json`. In both files, “q” represents the original queries of the datasets, while “q_short” represents the simplified queries.

Candidate cases of LeCaRD are located in `LeCaRD/data/candidates/similar_case` and candidate cases of CAIL2022 are located in `CAIL2022/stage2/candidates_stage2_valid`.

Labels of LeCaRD are located in `LeCaRD/data/label/label_top30_dict.json` and labels of CAIL2022 are located in `CAIL2022/label/label_1.json`.

#### Data Description

```
 [
 	{
            "uniqid": "...",
            "title": "...",
            "SS": "...",
            "query": "...",
            "key": "...",
            "AJAY": [..., ...],
            "AJDY": ...,
            "LY": "...",
            "JG": "...",
            "path": "...",
            "Main article": "['...', '...', '...']",
            "other article": "['...', '...', '...']",
            "sentence": "...",
            "similar case": "..."
    },
    ...
]
```

This section introduces only the important key names in the dataset.

-  **SS**: Description of the case, detailing the incident, the accused's actions, and the prosecution's claims. This section includes specific dates, locations, and circumstances surrounding the alleged crime. 
-  **query**: A brief summary of **SS**. When training the model, this section, along with **key**, will form the query-key pairs.
-  **key**: If the query and key of a pair are generated from the same case, the **key** section will be exactly the same as **SS**. Otherwise, the **key** section will be the same as **Similar Case**.
-  **AJAY**: Each number in this list represents a charge in this case. The correspondence between the numbers and the charges can be found in ```LEAD_data/mapping_v2.json```
-  **main article**: A list of relevant legal articles of *Chinese Criminal Law* referenced by this case. 
-  **other article**: An list of ancillary legal articles of *Chinese Criminal Law* referenced by this case. 
-  **sentence**: The length of the sentence imposed on the defendant. 
-  **similar case**: A detailed description of a case that shares similarities with the current case.