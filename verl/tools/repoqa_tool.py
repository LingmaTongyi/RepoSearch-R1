import logging
import os
import subprocess
from collections import defaultdict
from typing import Any, Optional, Union, List
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op
from verl.request_model.main import request_bailian_internal_models
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Error messages
TOOL_ERROR = {
    'exec_e': 'EXECUTION_ERROR',
    'format_e': 'FORMAT_ERROR'
}


def execute_bash_command_popen(command: Union[str, List[str]], timeout: Optional[int] = None, encoding: str = 'utf-8'):
    """Execute bash command using subprocess.Popen"""
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "exception": None,
    }
    
    try:
        # 创建进程，根据command类型决定是否使用shell
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=isinstance(command, str),  # 如果command是字符串，则使用shell=True
            text=False,  # 保持二进制输出，稍后手动解码
        )
        
        # 使用communicate获取输出，可以设置超时
        stdout, stderr = process.communicate(timeout=60)
        
        # 解码输出
        result["stdout"] = stdout.decode(encoding, errors='replace') if stdout else ""
        result["stderr"] = stderr.decode(encoding, errors='replace') if stderr else ""
        result["returncode"] = process.returncode
        result["success"] = process.returncode == 0  # 只有返回码为0才表示成功
        
    except subprocess.TimeoutExpired:
        # 处理超时情况
        process.kill()  # 终止进程
        result["stderr"] = "Tool executed timeout"
        result["returncode"] = -1  # 使用-1表示超时
        result["exception"] = f"Command execution timeout {timeout} seconds!"
        
    except Exception as e:
        result["stderr"] = f"An exception occurred during execution: {e}"
        result["exception"] = str(e)
    return result


class SearchKeywordTool(BaseTool):
    """Tool for searching keywords in folders."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None) or "/testbed",
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        keyword = parameters.get("keyword")
        folder_path = parameters.get("folder_path")
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        
        if not folder_path.startswith(project_root_path):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!", 0.0, {}
        
        if keyword is None or folder_path is None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for search_keyword_in_folder!", 0.0, {}
        
        # Search keyword with grep
        grep_command = f"grep --exclude-dir=.git --exclude-dir=.github --text -n -r '{keyword}' {folder_path} | head -n 200"
        result = execute_bash_command_popen(grep_command)
        
        if not result["success"]:
            if result["stderr"]:
                return f"{TOOL_ERROR.get('exec_e')}:Error searching for keyword '{keyword}': {result.get('stderr', '')}", 0.0, {}
            else:
                return f"No occurrences of keyword '{keyword}' found in the project!", 0.0, {}
        
        # Format the output
        raw_output = result["stdout"]
        fp_map_matches = defaultdict(lambda: 0)
        matches = raw_output.split('\n--\n')
        for match in matches:
            if match.strip():
                fp = match.strip().split(':')[0]
                fp_map_matches[fp] += 1
        
        format_output_str = ""
        for fp in fp_map_matches:
            format_output_str += f"Found {fp_map_matches[fp]} matches for keyword '{keyword}' in '{fp}' ({fp_map_matches[fp]} matches)\n"
        
        return f"The result of function calling is:\n{format_output_str}", 1.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class SearchFileTool(BaseTool):
    """Tool for searching files in folders."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None) or "/testbed",
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        file_name = parameters.get("file_name")
        folder_path = parameters.get("folder_path")
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        
        if file_name is None or folder_path is None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for search_file_in_folder!", 0.0, {}
        
        if not folder_path.startswith(project_root_path):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!", 0.0, {}
        
        # Construct the find command to search for the file
        find_command = f"find {folder_path} -name '{file_name}' -type f"
        
        # Execute the find command
        result = execute_bash_command_popen(find_command)
        
        if not result["success"]:
            return f"{TOOL_ERROR.get('exec_e')}: Failed to search for file '{file_name}' in path '{folder_path}', error: {result.get('stderr', '')}", 0.0, {}
        
        file_paths = result["stdout"].strip().split('\n')
        
        # Filter out empty lines
        file_paths = [path for path in file_paths if path]
        
        if not file_paths:
            return f"No files matching '{file_name}' found in path '{folder_path}'", 0.0, {}
        
        # Format the output
        output = f"Found {len(file_paths)} file(s) matching '{file_name}':\n"
        
        if len(file_paths) > 20:
            output += f"Only showing the top 20 results:\n"
            file_paths = file_paths[:20]
        for path in file_paths:
            output += f"- {path}\n"
        
        return f"The result of function calling is:\n{output}", 1.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ListFilesTool(BaseTool):
    """Tool for listing files in folders."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None) or "/testbed",
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        folder_path = parameters.get("folder_path")
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        
        if project_root_path and (not folder_path.startswith(project_root_path)):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!", 0.0, {}
        
        if folder_path is None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for list_files_in_folder!", 0.0, {}
        
        ls_command = f"ls {folder_path}"
        # Execute the ls command
        result = execute_bash_command_popen(ls_command)
        
        if not result["success"]:
            return f"{TOOL_ERROR.get('exec_e')}: An error occurred in list_files_in_folder while executing the command: {ls_command}, error: {result.get('stderr', '')}", 0.0, {}
        
        res = [f"{folder_path}/{fp}" for fp in result["stdout"].split('\n') if fp]
        res_str = '\n'.join(res)
        return f"The result of function calling is:\n{res_str}", 1.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ReviewFileTool(BaseTool):
    """Tool for reviewing file content."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None) or "/testbed",
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        file_path = parameters.get("file_path")
        start_line = parameters.get("start_lineno")
        end_line = parameters.get("end_lineno")
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        
        if not file_path.startswith(project_root_path):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!", 0.0, {}
        
        if file_path is None or start_line is None or end_line is None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for review_file!", 0.0, {}
        
        # Ensure start_line and end_line are integers
        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except ValueError:
            return f"{TOOL_ERROR.get('format_e')}: start_line and end_line must be integers", 0.0, {}
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            return f"File '{file_path}' does not exist!", 0.0, {}
        
        # Limit the number of lines to 100 to prevent excessive output
        if end_line - start_line > 100:
            end_line = start_line + 100
            note = f"\nYou can only review up to 100 lines of file content each round, but the tools call want to review {end_line - start_line} lines."
        else:
            note = ""
        
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Adjust line numbers (0-indexed in Python, 1-indexed in user input)
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            # Extract the requested lines
            requested_lines = lines[start_idx:end_idx]
            
            # Format the output with line numbers
            output = f"Content of '{file_path}' (lines {start_line} to {end_line}):\n\n"
            for i, line in enumerate(requested_lines, start=start_line):
                output += f"{i}| {line}"
            
            return f"The result of function calling is:\n{output + note}", 1.0, {}
        
        except Exception as e:
            return f"{TOOL_ERROR.get('exec_e')}:Error reading file: {str(e)}", 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class SearchSymbolTool(BaseTool):
    """Tool for searching symbols in files."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None) or "/testbed",
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        symbol = parameters.get("symbol")
        file_path = parameters.get("file_path")
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        
        if not file_path.startswith(project_root_path):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!", 0.0, {}
        
        if symbol is None or file_path is None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for search_symbol_in_file!", 0.0, {}
        
        if not os.path.isfile(file_path):
            return f"{TOOL_ERROR.get('exec_e')}: File '{file_path}' does not exist", 0.0, {}
        
        # Construct the grep command to search for the symbol
        grep_command = f"grep -n '{symbol}' '{file_path}'"
        
        # Execute the grep command
        result = execute_bash_command_popen(grep_command)
        
        if not result["success"]:
            # If grep returns non-zero exit code but it's just because no matches were found (exit code 1)
            if result["returncode"] == 1 and not result["stderr"]:
                return f"No occurrences of '{symbol}' found in '{file_path}'", 0.0, {}
            else:
                return f"{TOOL_ERROR.get('exec_e')}: Failed to search for '{symbol}' in '{file_path}'", 0.0, {}
        
        # Process the output to format it nicely
        matches = result["stdout"].strip().split('\n')
        
        # Format the output
        output = f"Found {len(matches)} occurrence(s) of '{symbol}' in '{file_path}':\n\n"
        for match in matches:
            output += f"{match}\n"
        
        return f"The result of function calling is:\n{output}", 1.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class SubmitAnswerTool(BaseTool):
    """Tool for submitting answers with LLM judge reward calculation."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Store LLM judge parameters from kwargs
        self._instance_dict[instance_id] = {
            "question": kwargs.get("question", ""),
            "question_related_code": kwargs.get("related_code", ""),
            "golden_answer": kwargs.get("ground_truth", ""),
            "enable_llm_judge": True,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        answer = parameters.get("answer", "")
        
        # Get instance data
        instance_data = self._instance_dict.get(instance_id, {})
        
        # Calculate reward using LLM judge if enabled
        reward = 0.0  # Default reward
        if instance_data.get("enable_llm_judge", True):
            try:
                reward = await self._calculate_llm_judge_reward(
                    candidate_answer=answer,
                    question=instance_data.get("question", ""),
                    question_related_code=instance_data.get("question_related_code", ""),
                    golden_answer=instance_data.get("golden_answer", "")
                )
                logger.info(f"LLM judge reward calculated: {reward}")
            except Exception as e:
                logger.error(f"Error calculating LLM judge reward: {e}")
                reward = 0.0
        
        return f"The result of function calling is:\n{answer}", reward, {"llm_judge_reward": reward}

    async def _calculate_llm_judge_reward(
        self, 
        candidate_answer: str, 
        question: str, 
        question_related_code: str, 
        golden_answer: str
    ) -> float:
        """Calculate reward using LLM judge."""
        
        # LLM judge prompt template
        TRAIN_LLM_JUDGE_SCORE = """
You are an impartial judge tasked with critically evaluating the quality of AI assistant responses to user questions.
You will be provided with:
1. A user question (possibly including code)
2. A reference answer
3. The AI assistant's answer

Begin your evaluation by thoroughly understanding the user question and reference answer,
and then rigorously assess the AI assistant's answer based on the following dimension: Completeness.

**IMPORTANT NOTE**: 
1. The reference answer may represent just one of many valid solutions. Evaluate based on factual correctness and effectiveness, even if the approach differs.
2. For questions involving code, pay special attention to both the explanation and code implementation.

After providing your detailed explanations, you must rate the answer on a strict scale of 1 to 100 in each of the following dimension.
You must focus solely on the specific criteria for each dimension when assigning a score:

## Score Dimensions

1. **Completeness**: Does the AI assistant's answer cover all aspects of the user question, or does it miss any critical points?
    - **Guidelines**:
        - Read the reference answer to identify all the key components that should be included in the response.
        - Carefully compare the AI assistant's answer with the reference answer to identify any missing critical points.
        - Ensure there are no crucial points omitted or overly brief explanations.
        - Consider whether the AI assistant's answer, even if using a different approach, addresses all aspects of the user's question.
    - **Scoring**:
        - **1-20**: Largely incomplete, many critical points missed; missing the majority of key aspects.
        - **21-40**: Significant omissions, partially complete; missing more than half of the key points.
        - **41-60**: Some omissions, but covers most key points; missing a few critical points.
        - **61-80**: Minor omissions, but mostly complete; missing one or two minor points.
        - **81-100**: Fully comprehensive, no points missed; covers all key aspects thoroughly.
    
You must adhere strictly to the response format to give the final verdict after evaluating the AI assistant's answer.
Response format is: 
## Judge's Evaluation
### **Completeness**:
[Your reasoning]
Final verdict is: [[Completeness: ?]].

## Input:
### User Question
{question}
{question_related_code}
### Reference Answer
{golden_answer}
### AI Assistant's Answer
{candidate_answer}
### Judge's Evaluation
"""
        
        try:
            # Format the prompt
            prompt = TRAIN_LLM_JUDGE_SCORE.format(
                question=question,
                question_related_code=question_related_code,
                golden_answer=golden_answer,
                candidate_answer=candidate_answer
            )
            
            # Call LLM judge API (you will implement this)
            response = request_bailian_internal_models(
                model_key='qwen2.5-max',
                prompt=prompt,
                messages=None,
                tools=None
            )
            
            # Parse the score from response
            reward = self._parse_judge_score(response['choices'][0]['message']['content'])
            return reward
            
        except Exception as e:
            logger.error(f"Error in LLM judge calculation: {e}")
            return 0.0
    
    def _parse_judge_score(self, response: str) -> float:
        """Parse the completeness score from LLM judge response."""
        try:
            import re
            
            # Look for pattern like "[[Completeness: 85]]"
            pattern = r"\[\[Completeness:\s*(\d+)\]\]"
            match = re.search(pattern, response)
            if match:
                score = int(match.group(1))
                if score < 1:
                    reward = 0.0
                elif score < 21:
                    reward = 0.2
                elif score < 41:
                    reward = 0.4
                elif score < 61:
                    reward = 0.6
                elif score < 81:
                    reward = 0.8
                elif score < 101:
                    reward = 1.0
                else:
                    logger.info(f"[parse_judge_score] score error: {score}")
                return reward
            else:
                logger.warning(f"Could not parse judge score from response: {response}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error parsing judge score: {e}")
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]



