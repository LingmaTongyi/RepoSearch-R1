import os
import re
import subprocess
import time
from typing import Dict, Any, Union, List, Optional
from collections import defaultdict
import logging
import sys
from uuid import uuid4

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



TOOL_ERROR = {
    'exec_e': '[Tool execution error]',
    'format_e': '[Tool calling format error]',
    'unknown_e': '[Error calling a non-existent tool]'
}

def execute_bash_command_popen(command: Union[str, List[str]], timeout: Optional[int] = None, encoding: str = 'utf-8'):
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
        # stdout, stderr = process.communicate(timeout=timeout)  # 获取已有输出
        # result["stdout"] = stdout.decode(encoding, errors='replace') if stdout else ""
        # result["stderr"] = stderr.decode(encoding, errors='replace') if stderr else ""
        result["stderr"] = "Tool excuted timeout"
        result["returncode"] = -1  # 使用-1表示超时
        result["exception"] = f"Command execution timeout {timeout} seconds!"
        
    except Exception as e:
        result["stderr"] = f"An exception occurred during execution: {e}"
        result["exception"] = str(e)
    return result

def parse_command(command):
    # Initialize result dictionary with default values
    result = {
        "command": None,
        "file_path": None,
        "start_line": None,
        "end_line": None,
        "file_name": None,
        "keyword": None,
        "symbol": None,
        "folder_path": None,
    }
    
    
    # Helper function to create regex pattern for quoted or unquoted parameters
    def param_pattern(flag):
        return f"{flag}\\s+(?:'([^']+)'|\"([^\"]+)\"|([^\\s-][^\\s]*))"
    
    # Command patterns dictionary: maps command names to their parameter patterns
    command_patterns = {
        "review_file": {
            "pattern": f"\\s*review_file\\s+{param_pattern('-f')}\\s+-s\\s+(\\d+)\\s+-e\\s+(\\d+)",
            "groups": [
                {"name": "file_path", "indices": [0, 1, 2]},
                {"name": "start_line", "index": 3, "type": int},
                {"name": "end_line", "index": 4, "type": int}
            ]
        },
        "search_file_in_folder": {
            "pattern": f"\\s*search_file_in_folder\\s+{param_pattern('-f')}\\s+{param_pattern('-p')}",
            "groups": [
                {"name": "file_name", "indices": [0, 1, 2]},
                {"name": "folder_path", "indices": [3, 4, 5]}
            ]
        },
        "search_symbol_in_file": {
            "pattern": f"\\s*search_symbol_in_file\\s+{param_pattern('-i')}\\s+{param_pattern('-f')}",
            "groups": [
                {"name": "symbol", "indices": [0, 1, 2]},
                {"name": "file_path", "indices": [3, 4, 5]}
            ]
        },
        "list_files_in_folder": {
            "pattern": f"\\s*list_files_in_folder\\s+(?:'([^']+)'|\"([^\"]+)\"|([^\\s]+))",
            "groups": [
                {"name": "folder_path", "indices": [0, 1, 2]}
            ]
        },
        "search_keyword_in_folder": {
            "pattern": f"\\s*search_keyword_in_folder\\s+{param_pattern('-k')}\\s+{param_pattern('-p')}",
            "groups": [
                {"name": "keyword", "indices": [0, 1, 2]},
                {"name": "folder_path", "indices": [3, 4, 5]}
            ]
        }
    }
    
    # Try to match each command pattern
    for cmd_name, cmd_config in command_patterns.items():
        match = re.search(cmd_config["pattern"], command)
        if match:
            result["command"] = cmd_name
            
            # Extract parameters based on group configuration
            for group in cmd_config["groups"]:
                if "indices" in group:  # Multiple possible matches (for quoted/unquoted values)
                    value = next((match.group(i+1) for i in group["indices"] if match.group(i+1) is not None), None)
                else:  # Single index
                    value = match.group(group["index"]+1)
                
                # Apply type conversion if specified
                if value is not None and "type" in group:
                    value = group["type"](value)
                
                result[group["name"]] = value
            
            break
    
    # If no command matched
    if result["command"] is None:
        return {"command": "unknown", "error": "No matching command pattern found"}
    
    return result

class search_keyword_in_folder:
    def __init__(self):
        self.description = "Search for a keyword in the folder and return file paths with the number of matches in each file."
        self.command_example = "search_keyword_in_folder -k 'keyword' -p 'folder_path'"
        self._instance_dict = {}
        
    def search_keyword_with_context(self,keyword, folder_path, context_lines=1):
        """
        Search for a symbol in the project directory and return file paths and context.
        
        Args:
            symbol_name (str): The name of the symbol to search for
            folder_path (str): The path to the project root directory
            context_lines (int): Number of lines to show before and after the match
            
        Returns:
            str: A formatted string with the search results including file paths and context
        """
        grep_command = f"grep  --exclude-dir=.git --exclude-dir=.github --text -n -A {context_lines} -B {context_lines}  -r '{keyword}' {folder_path} | head -n 200"
        
        raw_output = None

        result = execute_bash_command_popen(grep_command)

        if not result["success"]:
            if result["stderr"]:
                return f"{TOOL_ERROR.get('exec_e')}:Error searching for keyword '{keyword}': {result.get('stderr', '')}"
            else:
                return f"No occurrences of keyword '{keyword}' found in the project!"
        # Format the output
        raw_output = result["stdout"]
        return raw_output
    async def create(self, **kwargs) -> str:
        """Create a new tool instance."""
        instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None),
        }
        assert self._instance_dict[instance_id]["project_root_path"] is not None, "project_root_path is required!"
        return instance_id

    async def execute(self, instance_id, tool_call, **kwargs):
        # 原有的 runTool 逻辑
        params = parse_command(tool_call)
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        # Get parameters from the parsed command
        keyword = params.get('keyword', None)
        folder_path = params.get('folder_path', None)
        if not folder_path.startswith(project_root_path):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!"
        if keyword == None or folder_path == None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for search_keyword_in_project!"
        context_lines = 0
        raw_output = self.search_keyword_with_context(keyword, folder_path, context_lines)
        if not raw_output:
            return f"No occurrences of keyword '{keyword}' found in the project!"
        fp_map_matches = defaultdict(lambda: 0)
        matches = raw_output.split('\n--\n')
        for match in matches:
            fp = match.strip().split(':')[0]
            fp_map_matches[fp] += 1
        format_output_str = ""
        for fp in fp_map_matches:
            format_output_str += f"Found {fp_map_matches[fp]} matches for keyword '{keyword}' in '{fp}' ({fp_map_matches[fp]} matches)\n"
        return format_output_str

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class search_file_in_folder:
    def __init__(self):
        self.description = "Search the file within the folder scope. You need to provide the complete and exactly file name with suffix for searching like 'test_metric.cpp', 'setup.py'..."
        self.tool_name = "search_file_in_folder"
        self.command_example = "search_file_in_folder -f 'file_name' -p 'folder_path'"
        self._instance_dict = {}
    async def create(self, **kwargs) -> str:
        """Create a new tool instance."""
        instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None),
        }
        assert self._instance_dict[instance_id]["project_root_path"] is not None, "project_root_path is required!"
        return instance_id

    async def execute(self, instance_id, tool_call, **kwargs):
        # 原有的 runTool 逻辑
        params = parse_command(tool_call)
        file_name = params.get('file_name', None)
        folder_path = params.get('folder_path', None)
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        if file_name == None or folder_path == None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for search_file_in_folder!"
        if not folder_path.startswith(self._instance_dict[instance_id]["project_root_path"]):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!"
        # Construct the find command to search for the file
        find_command = f"find {folder_path} -name '{file_name}' -type f"
        
        # Execute the find command
        result = execute_bash_command_popen(find_command)
        
        if not result["success"]:
            return f"{TOOL_ERROR.get('exec_e')}: Failed to search for file '{file_name}' in path '{folder_path}', error: {result.get('stderr', '')}"
        
        file_paths = result["stdout"].strip().split('\n')
        
        # Filter out empty lines
        file_paths = [path for path in file_paths if path]
        
        if not file_paths:
            return f"No files matching '{file_name}' found in path '{folder_path}'"
        
        # Format the output
        # to avoid too many results, only show random 20 results
        output = f"Found {len(file_paths)} file(s) matching '{file_name}':\n"

        if len(file_paths) > 20:

            # file_paths = random.sample(file_paths, 20)
            output += f"Only showing the top 20 results:\n"
            file_paths = file_paths[:20]
        for path in file_paths:
            output += f"- {path}\n"
        
        return output

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
        

class list_files_in_folder:
    def __init__(self):
        self.description = "List all files within the folder."
        self.tool_name = "list_files_in_folder"
        self.command_example = "list_files_in_folder 'folder_path'"
        self._instance_dict = {}
    async def create(self, **kwargs) -> str:
        """Create a new tool instance."""
        instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None),
        }
        assert self._instance_dict[instance_id]["project_root_path"] is not None, "project_root_path is required!"
        return instance_id

    async def execute(self, instance_id, tool_call, **kwargs):
        params = parse_command(tool_call)
        # Get folder path from params
        folder_path = params.get('folder_path', None)
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        if not folder_path.startswith(self._instance_dict[instance_id]["project_root_path"]):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!"
        if folder_path == None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for list_files_in_folder!"
        ls_command = f"ls {folder_path}"
        # Execute the ls command
        result = execute_bash_command_popen(ls_command)
        
        if not result["success"]:
            return f"{TOOL_ERROR.get('exec_e')}: An error occurred in {self.tool_name} while executing the tool call: {tool_call}, error: {result.get('stderr', '')}"
        res = [f"{folder_path}/{fp}" for fp in result["stdout"].split('\n') if fp]
        return '\n'.join(res)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class review_file:
    def __init__(self):
        self.description = "View the file content between [start_lineno] and [end_lineno] in file. Each call can browse up to 100 lines of content. file_path is the absolute path of the file. Do not use this tool to find the code line by line! it is very inefficient. Not recommended if you do not know the exact location of the target code and the absolute file path!"
        self.tool_name = "review_file"
        self.command_example = "review_file -f 'file_path' -s start_lineno -e end_lineno"
        self._instance_dict = {}
    async def create(self, **kwargs) -> str:
        """Create a new tool instance."""
        instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None),
        }
        assert self._instance_dict[instance_id]["project_root_path"] is not None, "project_root_path is required!"
        return instance_id

    async def execute(self, instance_id, tool_call, **kwargs):
        params = parse_command(tool_call)
        
        # Get parameters from the parsed command
        file_path = params.get('file_path')
        start_line = params.get('start_line')
        end_line = params.get('end_line')
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        if not file_path.startswith(self._instance_dict[instance_id]["project_root_path"]):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!"

        if file_path == None or start_line == None or end_line == None:
            print(file_path, start_line, end_line)
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for review_file!"
        # Ensure start_line and end_line are integers
        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except ValueError:
            return f"{TOOL_ERROR.get('format_e')}: start_line and end_line must be integers"
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            return f"File '{file_path}' does not exist!"
        
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
            
            return output + note
        
        except Exception as e:
            return f"{TOOL_ERROR.get('exec_e')}:Error reading file: {str(e)}"

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
                    
        
class search_symbol_in_file:
    def __init__(self):
        self.description = "Search the specified symbol in the file. Not recommended if you do not know the exact location of the symbol. Notice this tool searches symbol in files instead of folders."
        self.command_example = "search_symbol_in_file -i 'symbol' -f 'file_path'"
        self._instance_dict = {}
    async def create(self, **kwargs) -> str:
        """Create a new tool instance."""
        instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "project_root_path": kwargs.get("project_root_path", None),
        }
        assert self._instance_dict[instance_id]["project_root_path"] is not None, "project_root_path is required!"
        return instance_id

    async def execute(self, instance_id, tool_call, **kwargs):
        params = parse_command(tool_call)  # Note: should be parse_command, not self.parse_command
        # Get parameters from the parsed command
        symbol = params.get('symbol', None)
        file_path = params.get('file_path', None)
        project_root_path = self._instance_dict[instance_id]["project_root_path"]
        if not file_path.startswith(self._instance_dict[instance_id]["project_root_path"]):
            return f"{TOOL_ERROR.get('exec_e')}: Folder path must start with [Absolute project root path]: {project_root_path}!"
        if symbol == None or file_path == None:
            return f"{TOOL_ERROR.get('format_e')}: Missing parameters for search_symbol_in_file!"
        if not os.path.isfile(file_path):
            return f"{TOOL_ERROR.get('exec_e')}: File '{file_path}' does not exist"
        
        # Construct the grep command to search for the symbol
        grep_command = f"grep -n '{symbol}' '{file_path}'"
        
        # Execute the grep command
        result = execute_bash_command_popen(grep_command)
        
        if not result["success"]:
            # If grep returns non-zero exit code but it's just because no matches were found (exit code 1)
            if result["returncode"] == 1 and not result["stderr"]:
                return f"No occurrences of '{symbol}' found in '{file_path}'"
            else:
                return f"{TOOL_ERROR.get('exec_e')}: Failed to search for '{symbol}' in '{file_path}'"
        
        # Process the output to format it nicely
        matches = result["stdout"].strip().split('\n')
        
        # Format the output
        output = f"Found {len(matches)} occurrence(s) of '{symbol}' in '{file_path}':\n\n"
        for match in matches:
            output += f"{match}\n"
        
        return output

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
TOOL_LIB = {
    "list_files_in_folder": list_files_in_folder(),
    "review_file": review_file(), # n~m 行
    "search_file_in_folder": search_file_in_folder(),# find_file
    "search_symbol_in_file": search_symbol_in_file(), # grep,
    "search_keyword_in_folder": search_keyword_in_folder()
}