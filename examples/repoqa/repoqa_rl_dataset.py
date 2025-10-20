from typing import Optional
import datasets
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.experimental.agent_loop.tools import TOOL_LIB
SYS_PROMPT = """You are an assistant helping a user answer a question about their project.\
Your goal is to search the code needed round by round so that you can answer their question finally. \
DO *NOT* try to modify the file or folder structure of the project. \
You have the following tools to search code in project and DO *NOT* use other tools.

# Tools: 
{tools_list}

You can perform multiple rounds of actions. In each round, you need to first think about the reasoning process and then call above tools in ```bash ... ``` block as Action. \
Tool call in ```bash ... ``` should not be empty each round. \
You can only make *ONE* Thought and *ONE* Action, like: 

### Thought: I need to search the keyword 'power' within folder '/abs/path/to/folder'.
### Action:
```bash
search_keyword_in_folder -k 'power' -p '/abs/path/to/folder'
```

DO *NOT* output any other texts before your Thought nor after your Action.
All the file or folder paths you provide in ```bash ... ``` block should be the absolute path.
DO *NOT* output the results of the action yourself. I will execute the action and give back the results of the action to you.


If you found that the conversation already included enough code snippets that you could answer the query perfectly, \
you can stop output '### Thought' and '### Action' and only output the final '### Answer' in the following format:

### Answer: The Answer to the question is ...
"""

FIRST_USER_PROMPT = """
# Input:
[Absolute project root path]: 
{project_root_path}
[Files and Folders within the project]:
{folder_structure}
[Question]: 
{question}
Here is my code(In my own space not in project):
{related_code}
"""
    
class RepoQARLHFDataset(RLHFDataset):

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.tool_str = ""
        for i, available_tool_name in enumerate(TOOL_LIB.keys()):
            self.tool_str += f"Tool '{available_tool_name}:' {TOOL_LIB[available_tool_name].description}\nTool Call example: {TOOL_LIB[available_tool_name].command_example}\n"
        super().__init__(data_files, tokenizer, config, processor)


    def _build_messages(self, row_dict: dict):
        
        project_root_path = row_dict['extra_info'].get('interaction_kwargs').get('project_root_path')
        related_code_str = row_dict['extra_info'].get('interaction_kwargs').get('related_code')

        init_messages = [
            {'role': 'system', 'content': SYS_PROMPT.format(tools_list=self.tool_str)},
            {'role': 'user', 'content': FIRST_USER_PROMPT.format(project_root_path=project_root_path, folder_structure=row_dict['extra_info'].get('interaction_kwargs').get('folder_structure'),question=row_dict['extra_info'].get('interaction_kwargs').get('question'), related_code=related_code_str)}
        ]
        return init_messages

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        
        # Create a function that replaces the prompt column with _build_messages result
        def replace_prompt_with_build_messages(example):
            # Get the new messages from _build_messages
            new_messages = self._build_messages(example)
            # Replace the prompt column with the new messages
            example["prompt"] = new_messages
            return example
        
        # Apply the function to replace prompt column values
        self.dataframe = self.dataframe.map(
            function=replace_prompt_with_build_messages,
            num_proc=self.num_workers,
        )

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)