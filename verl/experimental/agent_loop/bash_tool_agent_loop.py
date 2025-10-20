# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4
from datetime import datetime
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentCoTLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.experimental.agent_loop.tools import (
    parse_command,
    search_keyword_in_folder,
    search_file_in_folder,
    list_files_in_folder,
    review_file,
    search_symbol_in_file,

)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("bash_tool_agent")
class BashToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level bash_tool_agent initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.tools = {
            "list_files_in_folder": list_files_in_folder(),
            "review_file": review_file(), # n~m 行
            "search_file_in_folder": search_file_in_folder(),# find_file
            "search_symbol_in_file": search_symbol_in_file(), # grep,
            "search_keyword_in_folder": search_keyword_in_folder()
        }

        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True, enable_thinking=False)

        cls.traj_save_dir = config.actor_rollout_ref.rollout.multi_turn.traj_save_dir

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentCoTLoopOutput:
        # 初始化消息列表，直接维护结构化数据
        messages = list(kwargs["raw_prompt"])
        conversation_messages = copy.deepcopy(messages)  # 深拷贝维护完整的对话消息列表
        # 获取 extra_info 用于工具初始化
        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        metrics = {}
        request_id = uuid4().hex
        global_step = str(kwargs.get("step", -1))
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,enable_thinking=False
            ),
        )
        response_mask = []
        user_turns, assistant_turns = 0, 0
        # tool_rewards = []
        answer_reward = None
        while assistant_turns < self.max_assistant_turns:
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1
            
            # 提取工具调用
            
            # 解码assistant响应并提取工具调用
            assistant_response = await self.loop.run_in_executor(
                None, 
                lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            assistant_message = {
                'role': 'assistant',
                'content': assistant_response
            }
            conversation_messages.append(assistant_message)

            # reach max response length
            if len(response_mask) >= self.response_length:
                logger.info(f"Reach max response length and break: {self.response_length}")
                break

            tool_parse_message, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if tool_parse_message == 'finish':
                logger.info("Reach finish and break")
                answer, reward = await self.tool_parser.extract_answer_and_calculate_reward(
                    assistant_response,
                    interaction_kwargs.get("question", ""),
                    interaction_kwargs.get("related_code", ""),
                    interaction_kwargs.get("ground_truth", "")
                )
                answer_reward = reward  
                # tool_rewards.append(reward)
                break
            elif tool_parse_message == 'success':
                tool_response = "### Observation: "
                tool_call = tool_calls[0]
                # 调用工具
                tasks = []
                tasks.append(self._call_tool(tool_call, interaction_kwargs))
                with simple_timer("tool_calls", metrics):
                    task_results = await asyncio.gather(*tasks)
                    
                assert len(task_results) == 1, f"Expected 1 task result, got {len(task_results)}"
                result = task_results[0]
                if isinstance(result, Exception):
                    logger.error(f"Tool call error: {result}")
                    tool_response_content, tool_reward = f"Tool execution failed: {str(result)}", -1.0
                else:
                    tool_response_content, tool_reward = result
                tool_response += tool_response_content
                # tool_rewards.append(tool_reward)
            else:
                tool_response = "### Observation: "
                assert len(tool_calls) == 0, 'tool_calls should be empty'
                tool_response += tool_parse_message
                
            # 添加轮次限制提示
            cutdown_str = ""
            if self.max_assistant_turns - assistant_turns < 5:
                if self.max_assistant_turns - assistant_turns == 1:
                    cutdown_str = '\nThere is no search round left, please answer the question directly next round. Remember starting with "### Answer" !'
                else:
                    cutdown_str = f"\nOnly {self.max_assistant_turns - assistant_turns - 1} turns left to search."
            tool_response += cutdown_str
            tool_responses = [{
                'role': 'user',
                'content': tool_response
            }]
            
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                logger.info(f"Reach max response length and break: {self.response_length}")
                break

            conversation_messages.append(tool_responses[0])
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1
        
        response_ids = prompt_ids[-len(response_mask) :]
        # output_traj = self.loop.run_in_executor(None, self.tokenizer.decode, prompt_ids)
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        rm_scores = [0.0] * len(response_ids)
        if len(rm_scores) > self.response_length:
            rm_scores = rm_scores[: self.response_length]
        else:
            rm_scores[-1] = answer_reward if answer_reward is not None else 0.0
        

        output = AgentCoTLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            rm_scores=rm_scores,
            metrics=metrics,
        )
        
        # 保存轨迹数据到本地，直接使用维护的消息列表
        await self._save_trajectory(output, prompt_ids, response_ids, response_mask, answer_reward, interaction_kwargs, conversation_messages, global_step)
        
        # logger.info(f"The output of tool_agent_loop is: {output}")
        return output

    async def _save_trajectory(self, output: AgentCoTLoopOutput, prompt_ids: list[int], 
                              response_ids: list[int], response_mask: list[int], 
                              answer_reward: float, interaction_kwargs: dict = None,
                              conversation_messages: list = None, global_step=None) -> None:
        """保存agent采样的轨迹数据到本地文件。"""
        try:
            # 创建保存目录
            
            os.makedirs(self.traj_save_dir, exist_ok=True)
            # 从interaction_kwargs中获取instance_id
            instance_id = "unknown"
            if interaction_kwargs and 'instance_id' in interaction_kwargs:
                instance_id = interaction_kwargs['instance_id']
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"step_{global_step}_{timestamp}_{instance_id}.json"
            filepath = os.path.join(self.traj_save_dir, filename)
            
            assert conversation_messages is not None, "conversation_messages should not be None"
            
            # 构建轨迹数据
            trajectory_data = {
                "global_step": global_step,
                "num_turns": output.num_turns,
                "answer_reward": answer_reward,  # 保存原始列表
                "metrics": output.metrics.model_dump() if hasattr(output.metrics, 'model_dump') else str(output.metrics),
                "tokens": {
                    "prompt_length": len(prompt_ids),
                    "response_length": len(response_ids),
                },
                "messages": conversation_messages,  
            }
            
            # 异步写入文件
            def write_file():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
            
            await asyncio.get_event_loop().run_in_executor(None, write_file)
            
            logger.info(f"轨迹数据已保存到: {filepath}")
            logger.info(f"轨迹统计: {output.num_turns}轮对话, 总奖励: {answer_reward}")
            
        except Exception as e:
            logger.error(f"保存轨迹数据时出错: {e}")
            # 不抛出异常，避免影响主流程



    async def _call_tool(self, tool_call: str, interaction_kwargs: dict = None) -> tuple[str, float]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            params = parse_command(tool_call)
            tool_name = params["command"]
            if tool_name not in self.tools:
                logger.info(f"Tool NOT EXIST error, origin tool call is {tool_call}")
                return f'You use a not existing tool or use a wrong tool call foramt, please check your response. You can only use tools: {list(self.tools.keys())}\n.', -1.0
            
            tool = self.tools[tool_name]
            # 从 interaction_kwargs 中获取工具初始化参数
            create_kwargs = {}
            if interaction_kwargs:
                # 对于需要 project_root_path 的工具
                if "project_root_path" in interaction_kwargs:
                    create_kwargs["project_root_path"] = interaction_kwargs["project_root_path"]
            
            # 创建工具实例
            instance_id = await tool.create(**create_kwargs)
            # 执行工具，使用旧式调用方式（向后兼容）
            tool_response_raw = await tool.execute(instance_id, tool_call)
            tool_response = f'Tool {tool_name} search result:\n {tool_response_raw}\n'
            tool_reward = 1.0
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return str(e), -1.0
        finally:
            # 确保释放工具实例资源
            if tool and instance_id:
                await tool.release(instance_id)

        if len(tool_response) > self.max_tool_response_length:
            
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]
            logger.info(f"Tool {tool_name} response is too long, truncating to {len(tool_response)}")
        return tool_response, tool_reward
