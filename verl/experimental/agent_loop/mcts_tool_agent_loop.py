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
from datetime import datetime
import json
import logging
import math
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register, AgentCoTLoopOutput, AgentLoopMetrics
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
from collections import defaultdict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MCTSNode:
    """MCTS节点类，用于表示搜索树中的一个节点"""
    
    def __init__(self, id: str, parent: Optional['MCTSNode'] = None, depth: int = 0, 
                 chat_history: List[Dict] = None, prompt_ids: List[int] = None, 
                 response_mask: List[int] = None, children: List['MCTSNode'] = None,
                 n_visits: int = 0, q_value: float = 0.0, is_terminal: bool = False,
                 tool_call: str = "", tool_response: str = "", tool_reward: float = 0.0,
                 answer_reward: float = 0.0, response_ids: List[int] = None,
                 response_text: str = "", rollout_id: Optional[int] = None):
        self.id = id
        self.parent = parent
        self.depth = depth
        self.chat_history = chat_history if chat_history is not None else []
        self.prompt_ids = prompt_ids if prompt_ids is not None else []
        self.response_mask = response_mask if response_mask is not None else []
        self.children = children if children is not None else []
        self.n_visits = n_visits
        self.q_value = q_value
        self.is_terminal = is_terminal
        self.tool_call = tool_call
        self.tool_response = tool_response
        self.tool_reward = tool_reward
        self.answer_reward = answer_reward
        self.response_ids = response_ids if response_ids is not None else []
        self.response_text = response_text
        self.rollout_id = rollout_id


@register("mcts_tool_agent")
class MCTSToolAgentLoop(AgentLoopBase):
    """MCTS-based Tool Agent Loop with integrated Monte Carlo Tree Search."""

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level MCTSToolAgentLoop initialization")

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
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        
        # MCTS configuration from config if available
        mcts_config = config.actor_rollout_ref.rollout.get('mcts_agent', {})
        cls.num_rollouts = mcts_config.get('num_rollouts', 40)
        cls.min_reward_accumulation = mcts_config.get('min_reward_accumulation', 2)
        cls.max_depth_allowed = mcts_config.get('max_depth_allowed', 10)
        cls.max_children_num = mcts_config.get('max_children_num', 3)
        cls.exploration_weight = mcts_config.get('exploration_weight', 2.0)
        cls.weight_scheduler = mcts_config.get('weight_scheduler', 'exp')
        cls.try_times = mcts_config.get('try_times', 1)
        cls.enable_reflection = mcts_config.get('enable_reflection', False)
        cls.traj_save_dir = mcts_config.get('traj_save_dir', '/mnt/lgc/lgc/verl/mcts_traj_output')

    def _create_node_id(self) -> str:
        """Generate unique node ID using UUID."""
        return uuid4().hex

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        return self.exploration_weight * (0.1 ** (rollout_id / self.num_rollouts))
    def _calculate_uct_score(self, node: MCTSNode, rollout_id: int) -> float:
        """Calculate UCT score for node selection."""
        if node.n_visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        # Check if parent exists (root node has no parent)
        if node.parent is None:
            return node.q_value / node.n_visits  # Just return average Q value for root
        weight = self._get_weight(rollout_id)
        # Standard UCT formula: Q/N + c * sqrt(ln(parent_N) / N)
        exploitation = node.q_value / node.n_visits  # Average reward
        exploration = weight * math.sqrt(math.log(node.parent.n_visits) / node.n_visits)
        return exploitation + exploration

    @rollout_trace_op
    async def _generate_children(self, node: MCTSNode, sampling_params: Dict[str, Any], interaction_kwargs: Dict[str, Any], metrics: Dict[str, Any], request_id: str, rollout_id: int, enable_reflection: bool, child_1_node=None) -> Optional[MCTSNode]:
        
        # inherit parent's values - 使用安全的拷贝方式避免递归
        child_prompt_ids = list(node.prompt_ids) if node.prompt_ids else []
        child_response_mask = list(node.response_mask) if node.response_mask else []
        child_response_ids = None
        child_chat_history = deepcopy(node.chat_history)

        reflection_msg = None
        if enable_reflection:
            reflection_msg = deepcopy(child_1_node.chat_history)
            if reflection_msg[-1]['role'] == 'user':
                reflection_msg[-1]['content'] += '\nWait! Maybe you made some mistakes! You need to rethink the last round "### Thought" and "### Action" and try another response.'
            else:
                # reach the final answer
                reflection_msg.append({
                    "role": "user",
                    "content": '\nWait! Maybe you made some mistakes! You need to rethink and try another answer again, remember starting with "### Answer" Tag!.'
                })

        child_is_terminal = False
        child_answer_reward = None
        child_tool_response = ""
        child_response_text = ""
        child_tool_call = ""
        child_tool_reward = 0.0
        child_depth = node.depth + 1

        if child_depth >= self.max_depth_allowed:
            child_is_terminal = True
        """Generate child nodes using LLM inference."""
        try:
            # Prepare prompt from chat history
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    node.chat_history if not enable_reflection else reflection_msg, add_generation_prompt=True, tokenize=True, enable_thinking=False
                ),
            )
            
            # Generate response
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )

            child_response_ids = response_ids

            child_prompt_ids += response_ids
            child_response_mask += [1] * len(response_ids)
            
            # Decode response
            child_response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            
            # Create new chat history with assistant response
            child_chat_history.append({
                "role": "assistant",
                "content": child_response_text
            })
            
            tool_parse_message, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if tool_parse_message == 'finish':
                child_is_terminal = True
                answer, child_answer_reward = await self.tool_parser.extract_answer_and_calculate_reward(
                    child_response_text,
                    interaction_kwargs.get("question", ""),
                    interaction_kwargs.get("related_code", ""),
                    interaction_kwargs.get("ground_truth", "")
                )
            else:
                if tool_parse_message == 'success':
                    child_tool_response = "### Observation: "
                    child_tool_call = tool_calls[0]
                    # 调用工具
                    tasks = []
                    tasks.append(self._call_tool(child_tool_call, interaction_kwargs))
                    with simple_timer("tool_calls", metrics):
                        task_results = await asyncio.gather(*tasks)
                        
                    assert len(task_results) == 1, f"Expected 1 task result, got {len(task_results)}"
                    result = task_results[0]
                    if isinstance(result, Exception):
                        logger.error(f"Tool call error: {result}")
                        tool_response, tool_reward = f"Tool execution failed: {str(result)}", -1.0
                    else:
                        tool_response, tool_reward = result
                    child_tool_reward = tool_reward
                    child_tool_response += tool_response
                else:
                    assert len(tool_calls) == 0, 'tool_calls should be empty'
                    child_tool_reward = -1.0
                    child_tool_response += tool_parse_message
                cutdown_str = ""
                if self.max_depth_allowed - child_depth < 5:
                    if self.max_depth_allowed - child_depth == 1:
                        cutdown_str = '\nThere is no search round left, please answer the question directly next round. Remember starting with "### Answer" !'
                    else:
                        cutdown_str = f"\nOnly {self.max_depth_allowed - child_depth - 1} turns left to search."
                child_tool_response += cutdown_str
                tool_responses = [{
                    'role': 'user',
                    'content': child_tool_response
                }]
                child_chat_history.append(tool_responses[0])
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
                    ),
                )
                tool_response_ids = tool_response_ids[len(self.system_prompt) :]
                if len(child_prompt_ids) + len(tool_response_ids) >= self.response_length:
                    logger.info(f"Reach max response length and set is_terminal: {self.response_length}")
                    child_is_terminal = True
                child_prompt_ids += tool_response_ids
                child_response_mask += [0] * len(tool_response_ids)
            
            assert  child_tool_reward >= -1.0 and child_tool_reward <= 1.0, f"Invalid tool reward: {child_tool_reward}, tool_response is {child_tool_response}"
            if child_answer_reward:
                assert  child_answer_reward >= 0.0 and child_answer_reward <= 1.0, f"Invalid answer reward: {child_answer_reward}, answer is {child_response_text}"
            child = MCTSNode(
                id=self._create_node_id(),
                parent=node,
                depth=child_depth,
                chat_history=child_chat_history,
                prompt_ids=child_prompt_ids,
                response_mask=child_response_mask,
                response_text=child_response_text,
                rollout_id=rollout_id,
                tool_response=child_tool_response,
                tool_reward=child_tool_reward,
                is_terminal=child_is_terminal,
                tool_call=child_tool_call,
                answer_reward=child_answer_reward,
                response_ids=child_response_ids
            )
        
            return child
            
        except Exception as e:
            logger.error(f"Error generating children: {e}")
            return None

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        # Initialize MCTS rollout
        messages = list(kwargs["raw_prompt"])
        conversation_messages = copy.deepcopy(messages)
        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        metrics = {}
        metrics['trajectory_reward'] = 0.0
        explored_nodes = set()
        request_id = uuid4().hex
        global_step = str(kwargs.get("step", -1))

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                conversation_messages, add_generation_prompt=False, tokenize=True, enable_thinking=False
            ),
        )
        
        # Initialize MCTS tree using MCTSNode class
        root = MCTSNode(
            id=self._create_node_id(),
            parent=None,
            depth=0,
            chat_history=conversation_messages,
            prompt_ids=prompt_ids,
            rollout_id=None
        )
        nodes_by_id = {root.id: root}
        # breakpoint()

        
        logger.info(f"Starting MCTS rollout with {self.num_rollouts} iterations")

        async def _find_children(node: MCTSNode, sampling_params: Dict[str, Any], interaction_kwargs: Dict[str, Any], metrics: Dict[str, Any], request_id: str, rollout_id: int, max_children_num: int):
            children_list = []
            # assume max_children_num is 2, we only add reflection prompt for the second child
            # no reflection prompt for the first child
            child_1 = await self._generate_children(node, sampling_params, interaction_kwargs, metrics, request_id, rollout_id, enable_reflection=False)
            assert child_1 is not None, 'Failed to generate child node'
            children_list.append(child_1)
            nodes_by_id[child_1.id] = child_1
            
            # Generate reflection child based on child_1
            if max_children_num > 1:
                child_2 = await self._generate_children(node, sampling_params, interaction_kwargs, metrics, request_id, rollout_id, enable_reflection=True, child_1_node=child_1)
                if child_2 is not None:
                    children_list.append(child_2)
                    nodes_by_id[child_2.id] = child_2
                else:
                    logger.warning("Failed to generate reflection child node")
            
            node.children = children_list
        
        # MCTS main loop
        for iteration in range(self.num_rollouts):
            try:
                logger.info(f"MCTS iteration {iteration + 1}/{self.num_rollouts}")
                # SELECTION: Navigate from root to leaf using UCT
                current_node = root
                path = []
                while True:
                    path.append(current_node)
                        
                    if not current_node.children:
                        break
                    
                    # Find unexplored children
                    unexplored_children = [child for child in current_node.children if child not in explored_nodes]
                    
                    if unexplored_children:
                        current_node = random.choice(unexplored_children)
                        path.append(current_node)
                        break
                    
                    # Select best child using UCT
                    best_child = max(current_node.children, key=lambda child: self._calculate_uct_score(child, iteration))
                    current_node = best_child
                
                leaf_node = path[-1]  
                # --- EXPAND ---

                if leaf_node.is_terminal:
                    logger.info(f"[EXPAND] current node is a terminal node, no need to expand")
                else:
                    await _find_children(leaf_node, sampling_params, interaction_kwargs, metrics, request_id, iteration, self.max_children_num)
                
                
                # SIMULATION: Random rollout from current node (simplified)
                simu_path = []
                simu_node = leaf_node

                while not simu_node.is_terminal:
                    # Simple simulation: just check if we can generate more children
                    if not simu_node.children:
                        await _find_children(simu_node, sampling_params, interaction_kwargs, metrics, request_id, iteration, self.max_children_num)
                    simu_node = random.choice(simu_node.children)
                    simu_path.append(simu_node)
                full_path = path + simu_path
                final_node = full_path[-1]
                # Use tool_reward instead of tool_rewards list


                reward = final_node.answer_reward
                # BACKPROPAGATION: Update all nodes in path
                if final_node.n_visits == 0:
                    for node in reversed(full_path):
                        final_node.tool_reward += node.tool_reward 
                        node.q_value += reward
                        node.n_visits += 1
                        explored_nodes.add(node)
                else:
                    for node in reversed(full_path):
                        node.q_value += reward
                        node.n_visits += 1
                        explored_nodes.add(node)
            except Exception as e:
                logger.error(f"Error in MCTS iteration {iteration}: {e}")
                continue
        
        
        logger.info(f"MCTS completed. Tree size: {len(explored_nodes)} nodes")

        leaf_nodes = [node for node in explored_nodes if node.is_terminal]
        for node in leaf_nodes:
            node.answer_reward = node.answer_reward + 0.1 * node.tool_reward / node.depth
        
        # 保留8条轨迹，确保不同reward值至少一条，剩余的保证reward分布均衡
        from collections import defaultdict
        
        # 按reward值分组
        reward_groups = defaultdict(list)
        for node in leaf_nodes:
            reward_key = node.answer_reward
            reward_groups[reward_key].append(node)
        
        filtered_leaf_nodes = []
        target_count = 8
        
        # 获取所有unique reward值并按降序排序
        unique_rewards = list(reward_groups.keys())
        unique_rewards.sort(reverse=True)  # 按reward值降序排序
        
        # 首先确保选择最大和最小reward的节点
        if len(unique_rewards) > 0:
            # 选择最大reward的节点
            max_reward = unique_rewards[0]  # 已按降序排序，第一个是最大的
            filtered_leaf_nodes.append(reward_groups[max_reward][0])
            
            # 如果有不同的reward值，选择最小reward的节点
            if len(unique_rewards) > 1:
                min_reward = unique_rewards[-1]  # 最后一个是最小的
                filtered_leaf_nodes.append(reward_groups[min_reward][0])
        
        # 然后确保其他不同的reward值也至少选择一个节点
        for reward in unique_rewards:
            if len(filtered_leaf_nodes) < target_count:
                # 跳过已经选择过的最大和最小reward
                if reward == unique_rewards[0] or (len(unique_rewards) > 1 and reward == unique_rewards[-1]):
                    continue
                filtered_leaf_nodes.append(reward_groups[reward][0])
        
        # 如果还没达到target_count，从最高和最低reward交替选择剩余节点
        if len(filtered_leaf_nodes) < target_count:
            left_idx = 0  # 指向最高reward
            right_idx = len(unique_rewards) - 1  # 指向最低reward
            select_high = True  # 标记是否选择高reward
            
            while len(filtered_leaf_nodes) < target_count:
                if select_high and left_idx < len(unique_rewards):
                    # 从最高reward组选择下一个节点
                    reward = unique_rewards[left_idx]
                    remaining_nodes = reward_groups[reward][1:]  # 跳过已选择的第一个
                    if remaining_nodes:
                        filtered_leaf_nodes.append(remaining_nodes[0])
                        reward_groups[reward] = remaining_nodes[1:]  # 更新剩余节点
                    else:
                        left_idx += 1  # 该reward组没有更多节点，移到下一个
                        continue
                elif not select_high and right_idx >= 0:
                    # 从最低reward组选择下一个节点
                    reward = unique_rewards[right_idx]
                    remaining_nodes = reward_groups[reward][1:]  # 跳过已选择的第一个
                    if remaining_nodes:
                        filtered_leaf_nodes.append(remaining_nodes[0])
                        reward_groups[reward] = remaining_nodes[1:]  # 更新剩余节点
                    else:
                        right_idx -= 1  # 该reward组没有更多节点，移到下一个
                        continue
                else:
                    # 没有更多节点可选择
                    break
                
                # 切换选择策略
                select_high = not select_high
        
        logger.info(f"Filtered leaf nodes from {len(leaf_nodes)} to {len(filtered_leaf_nodes)} (kept 8 trajectories with balanced reward distribution)")
        if len(reward_groups) > 0:
            reward_distribution = {reward: len([n for n in filtered_leaf_nodes if (n.answer_reward if n.answer_reward is not None else 0.0) == reward]) for reward in reward_groups.keys()}
            logger.info(f"Final reward distribution: {reward_distribution}")
        leaf_nodes = filtered_leaf_nodes

        mcts_output_trajs = []

        for leaf_node in leaf_nodes:
            # Extract response_ids using response_mask length from the end of prompt_ids
            # prompt_ids contains full conversation, response_mask marks all tokens after first 2 rounds
            response_ids = leaf_node.prompt_ids[-len(leaf_node.response_mask) :]
            prompt_ids = leaf_node.prompt_ids[: len(leaf_node.prompt_ids) - len(leaf_node.response_mask)]
                
            rm_scores = [0.0] * len(response_ids)
            if len(rm_scores) > self.response_length:
                rm_scores = rm_scores[: self.response_length]
            elif len(rm_scores) > 0:  # Only set if we have scores
                # Safely handle None answer_reward
                rm_scores[-1] = leaf_node.answer_reward if leaf_node.answer_reward is not None else 0.0
            else:
                logger.warning(f"Leaf node {leaf_node.id} has no response_ids, skipping")
                
            # Add trajectory reward to metrics
            trajectory_reward = leaf_node.answer_reward if leaf_node.answer_reward is not None else 0.0
            metrics['trajectory_reward'] += trajectory_reward
            
            output = AgentCoTLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=leaf_node.response_mask[: self.response_length],
                num_turns=len(leaf_node.chat_history),
                rm_scores=rm_scores,
                metrics=metrics,
            )
            mcts_output_trajs.append(output)
        logger.info(f"MCTS tool agent loop output {len(mcts_output_trajs)} trajs")
        
        await self._save_trajectory(mcts_output_trajs, leaf_nodes, root, interaction_kwargs['instance_id'], global_step)
        
        return mcts_output_trajs

    async def _save_trajectory(self, mcts_output_trajs: List[AgentCoTLoopOutput], leaf_nodes: List[MCTSNode], root_node: MCTSNode, instance_id: str, global_step: str) -> None:
        """保存MCTS采样的多条轨迹数据到本地文件，包含chat_history和按层级组织的所有节点。"""
        try:
            # 创建保存目录
            os.makedirs(self.traj_save_dir, exist_ok=True)
            
            # 生成唯一文件名，格式为 mcts_{timestamp}_{instance_id}_{global_step}.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"step_{global_step}_{timestamp}_{instance_id}.json"
            filepath = os.path.join(self.traj_save_dir, filename)
            
            # 收集所有节点并按层级组织
            def collect_all_nodes_by_layer(root: MCTSNode) -> Dict[int, List[MCTSNode]]:
                """递归收集所有节点并按深度（层级）分组"""
                nodes_by_layer = {}
                
                def traverse(node: MCTSNode):
                    depth = node.depth
                    if depth not in nodes_by_layer:
                        nodes_by_layer[depth] = []
                    nodes_by_layer[depth].append(node)
                    
                    # 递归遍历子节点
                    for child in node.children:
                        traverse(child)
                
                traverse(root)
                return nodes_by_layer
            
            nodes_by_layer = collect_all_nodes_by_layer(root_node)
            
            # 构建按层级组织的节点数据
            layers_data = {}
            total_nodes = 0
            for depth, nodes in sorted(nodes_by_layer.items()):
                layer_nodes = []
                for node in nodes:
                    node_data = {
                        "node_id": node.id,
                        "parent_id": node.parent.id if node.parent else None,
                        "children_ids": [child.id for child in node.children],
                        "depth": node.depth,
                        "n_visits": node.n_visits,
                        "q_value": node.q_value,
                        "is_terminal": node.is_terminal,
                        "tool_call": node.tool_call,
                        "tool_response": node.tool_response,
                        "tool_reward": node.tool_reward,
                        "answer_reward": node.answer_reward,
                        "rollout_id": node.rollout_id,
                        "chat_history": node.chat_history,
                        "prompt_ids_length": len(node.prompt_ids) if node.prompt_ids else 0,
                        "response_mask_length": len(node.response_mask) if node.response_mask else 0,
                    }
                    layer_nodes.append(node_data)
                
                layers_data[f"layer_{depth}"] = {
                    "depth": depth,
                    "node_count": len(layer_nodes),
                    "nodes": layer_nodes
                }
                total_nodes += len(layer_nodes)
            
            # 构建叶子节点轨迹数据（保持原有格式）
            trajectories_data = []
            for i, (output, leaf_node) in enumerate(zip(mcts_output_trajs, leaf_nodes)):
                trajectory_data = {
                    "trajectory_id": f"mcts_traj_{i}",
                    "node_id": leaf_node.id,
                    "num_turns": output.num_turns,
                    "metrics": output.metrics if isinstance(output.metrics, dict) else {},
                    
                    # 对话历史数据
                    "chat_history": leaf_node.chat_history,
                    
                    # MCTS节点信息
                    "mcts_info": {
                        "depth": leaf_node.depth,
                        "n_visits": leaf_node.n_visits,
                        "q_value": leaf_node.q_value,
                        "is_terminal": leaf_node.is_terminal,
                        "tool_call": leaf_node.tool_call,
                        "tool_response": leaf_node.tool_response,
                        "tool_reward": leaf_node.tool_reward,
                        "answer_reward": leaf_node.answer_reward,
                        "rollout_id": leaf_node.rollout_id,
                    },
                    
                    # Token长度数据
                    "tokens": {
                        "prompt_length": len(output.prompt_ids),
                        "response_length": len(output.response_mask),
                    },
                    
                    # 统计信息
                    "statistics": {
                        "llm_generated_tokens": sum(output.response_mask) if output.response_mask else 0,
                        "tool_response_tokens": len(output.response_mask) - sum(output.response_mask) if output.response_mask else 0,
                        "total_tokens": len(output.prompt_ids) + len(output.response_ids),
                    }
                }
                trajectories_data.append(trajectory_data)
            
            # 构建完整的MCTS数据
            mcts_data = {
                "mcts_session_id": instance_id,
                "total_trajectories": len(mcts_output_trajs),
                "total_nodes": total_nodes,
                "max_depth": max(nodes_by_layer.keys()) if nodes_by_layer else 0,
                
                # 按层级组织的所有节点数据
                "tree_structure": {
                    "layers": layers_data,
                    "layer_summary": {
                        f"layer_{depth}": len(nodes) 
                        for depth, nodes in nodes_by_layer.items()
                    }
                },
                
                # 叶子节点轨迹数据（向后兼容）
                "trajectories": trajectories_data,
            }
            
            # 异步写入文件
            def write_file():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(mcts_data, f, ensure_ascii=False, indent=2)
            
            await asyncio.get_event_loop().run_in_executor(None, write_file)
            
            logger.info(f"MCTS轨迹数据已保存到: {filepath}")
            logger.info(f"MCTS统计: 共{len(mcts_output_trajs)}条轨迹, {total_nodes}个节点, 最大深度{max(nodes_by_layer.keys()) if nodes_by_layer else 0}")
            logger.info(f"各层节点数: {dict(sorted([(depth, len(nodes)) for depth, nodes in nodes_by_layer.items()]))}")
            
        except Exception as e:
            logger.error(f"保存MCTS轨迹数据时出错: {e}")
            # 不抛出异常，避免影响主流程



    async def _call_tool(self, tool_call: str, interaction_kwargs: dict = None) -> tuple[str, float]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            params = parse_command(tool_call)
            tool_name = params["command"]
            if tool_name not in self.tools:
                logger.info(f"Tool NOT EXIST error, LLM is using {tool_call}")
                return f'Tool NOT EXIST or ERROR Tool format!!! You can only use tools: {list(self.tools.keys())}\n, and check the foramt of your tool call.', -1.0
            
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
