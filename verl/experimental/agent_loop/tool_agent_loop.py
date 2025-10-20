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

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentCoTLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentCoTLoopOutput:
        # 初始化消息列表，直接维护结构化数据
        messages = list(kwargs["raw_prompt"])
        conversation_messages = copy.deepcopy(messages)  # 深拷贝维护完整的对话消息列表
        # 获取 extra_info 用于工具初始化
        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []
        is_finish = False
        user_turns, assistant_turns = 0, 0
        tool_rewards = []
        while True:
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            response_ids = [token_id for token_id in response_ids if token_id != 151644]
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1
            
            # reach max response length
            if len(response_mask) >= self.response_length:
                logger.info(f"Reach max response length and break: {self.response_length}")
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                logger.info(f"Reach max assistant turns and break: {self.max_assistant_turns}")
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                logger.info(f"Reach max user turns and break: {self.max_user_turns}")
                break

            # 提取工具调用
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            # 解码assistant响应并提取工具调用
            assistant_response = await self.loop.run_in_executor(
                None, 
                lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            # 构建 assistant 消息
            assistant_message = {
                'role': 'assistant',
                'content': assistant_response
            }
            tool_response = ""
            if not tool_calls:
                # 没有工具调用，添加错误提示
                tool_response = "Your response does not follow the correct format for calling the tool, or you did not call the tool. Please check your response and try again."
                # breakpoint()
                # 添加不包含tool_calls的assistant消息
                conversation_messages.append(assistant_message)
            elif len(tool_calls) > 1:
                # 多个工具调用，添加错误提示
                tool_response = "You can only call one tool at a time. Please check your response and try again."
                # 添加不包含tool_calls的assistant消息
                conversation_messages.append(assistant_message)
            else:
                # 正常的工具调用，添加tool_calls到assistant消息
                tool_call = tool_calls[0]
                assistant_message['tool_calls'] = [{
                    'type': 'function',
                    'function': {
                        'name': tool_call.name,
                        'arguments': tool_call.arguments
                    }
                }]
                conversation_messages.append(assistant_message)
                
                # 调用工具
                tasks = []
                tasks.append(self._call_tool(tool_call, interaction_kwargs))
                
                with simple_timer("tool_calls", metrics):
                    task_results = await asyncio.gather(*tasks)
                    
                # 处理工具响应
                assert len(task_results) == 1, f"Expected 1 task result, got {len(task_results)}"
                result = task_results[0]
                if isinstance(result, Exception):
                    logger.info(f"Tool call error and break")
                    break
                tool_response, tool_reward = result
                if tool_call.name == "submit_answer":
                    is_finish = True
                    tool_rewards.append(tool_reward)
            
            if is_finish:
                break
                
            # 添加轮次限制提示
            cutdown_str = ""
            if self.max_assistant_turns - assistant_turns < 5:
                if self.max_assistant_turns - assistant_turns == 1:
                    cutdown_str = "\nThere is no search round left, please answer the question directly next round. Please use the `submit_answer` tool to submit your answer."
                else:
                    cutdown_str = f"\nOnly {self.max_assistant_turns - assistant_turns - 1} turns left to search."
            tool_response += cutdown_str
            tool_responses = [{
                'role': 'tool',
                'content': tool_response
            }]
            
            # 将tool响应添加到对话消息列表
            conversation_messages.append(tool_responses[0])
            
            # logger.info(f"Tool calls tool_responses: {tool_responses}")
            # logger.info(f"Tool calls tool_rewards: {tool_rewards}")
            # append tool_response_ids
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                logger.info(f"Reach max response length and break: {self.response_length}")
                break

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
            rm_scores[-1] = tool_rewards[0] if len(tool_rewards) > 0 else 0.0
        

        output = AgentCoTLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            rm_scores=rm_scores,
            metrics=metrics,
        )
        
        # 保存轨迹数据到本地，直接使用维护的消息列表
        await self._save_trajectory(output, prompt_ids, response_ids, response_mask, tool_rewards, interaction_kwargs, conversation_messages)
        
        # logger.info(f"The output of tool_agent_loop is: {output}")
        return output

    async def _save_trajectory(self, output: AgentCoTLoopOutput, prompt_ids: list[int], 
                              response_ids: list[int], response_mask: list[int], 
                              tool_rewards: list[float], interaction_kwargs: dict = None,
                              conversation_messages: list = None) -> None:
        """保存agent采样的轨迹数据到本地文件。"""
        try:
            # 创建保存目录
            save_dir = "./trajectory_logs"
            os.makedirs(save_dir, exist_ok=True)
            
            # 从interaction_kwargs中获取instance_id
            instance_id = "unknown"
            if interaction_kwargs and 'instance_id' in interaction_kwargs:
                instance_id = interaction_kwargs['instance_id']
            
            # 生成唯一文件名，格式为 {instance_id}_{trajectory_id}.json
            trajectory_id = str(uuid4())[:8]
            filename = f"{instance_id}_{trajectory_id}.json"
            filepath = os.path.join(save_dir, filename)
            
            # 使用传入的conversation_messages，如果没有则使用解析逻辑（向后兼容）
            if conversation_messages is not None:
                # 直接使用维护的消息列表
                parsed_messages = conversation_messages
            else:
                # 回退到原有的解码逻辑（向后兼容）
                try:
                    if prompt_ids and response_ids:
                        full_text = self.tokenizer.decode(prompt_ids + response_ids, skip_special_tokens=True)
                        # 直接保存原始文本，不再进行解析
                        parsed_messages = full_text
                    else:
                        parsed_messages = []
                except Exception as decode_error:
                    logger.warning(f"解码token时出错: {decode_error}")
                    parsed_messages = []
            
            # 构建轨迹数据
            trajectory_data = {
                "trajectory_id": trajectory_id,
                "num_turns": output.num_turns,
                "tool_rewards": tool_rewards,  # 保存原始列表
                "metrics": output.metrics.model_dump() if hasattr(output.metrics, 'model_dump') else str(output.metrics),
                
                # Token长度数据
                "tokens": {
                    "prompt_length": len(prompt_ids),
                    "response_length": len(response_ids),
                },
                
                # 结构化消息数据
                "messages": parsed_messages,  # 结构化消息列表
                
                # 统计信息
                "statistics": {
                    "llm_generated_tokens": sum(response_mask) if response_mask else 0,
                    "tool_response_tokens": len(response_mask) - sum(response_mask) if response_mask else 0,
                    "total_tokens": len(prompt_ids) + len(response_ids),
                }
            }
            
            # 异步写入文件
            def write_file():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
            
            await asyncio.get_event_loop().run_in_executor(None, write_file)
            
            logger.info(f"轨迹数据已保存到: {filepath}")
            logger.info(f"轨迹统计: {output.num_turns}轮对话, 总奖励: {sum(tool_rewards) if tool_rewards else 0.0}")
            
        except Exception as e:
            logger.error(f"保存轨迹数据时出错: {e}")
            # 不抛出异常，避免影响主流程



    async def _call_tool(self, tool_call: FunctionCall, interaction_kwargs: dict = None) -> tuple[str, float]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            # 从 interaction_kwargs 中获取工具初始化参数
            create_kwargs = {}
            if interaction_kwargs:
                # 对于需要 project_root_path 的工具
                if "project_root_path" in interaction_kwargs:
                    create_kwargs["project_root_path"] = interaction_kwargs["project_root_path"]
                
                # 对于 SubmitAnswerTool，传递 LLM judge 所需参数
                if tool_name == "submit_answer":
                    if "question" in interaction_kwargs:
                        create_kwargs["question"] = interaction_kwargs["question"]
                    if "related_code" in interaction_kwargs:
                        create_kwargs["question_related_code"] = interaction_kwargs["related_code"]
                    if "ground_truth" in interaction_kwargs:
                        create_kwargs["golden_answer"] = interaction_kwargs["ground_truth"]
            
            # logger.info(f"Calling tool '{tool_name}' with create_kwargs: {list(create_kwargs.keys())}")
            instance_id = await tool.create(**create_kwargs)
            tool_response, tool_reward, _ = await tool.execute(instance_id, tool_args)
            # if tool_name == "submit_answer":
            #     logger.info(f"Tool response: {tool_response}")
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
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
