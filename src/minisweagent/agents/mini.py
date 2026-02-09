"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0
    observation_reasoning_template: str = (
        "The following observation is very long. Please analyze it and provide a concise summary "
        "focusing on the key information relevant to completing the task. Include:\n"
        "1. What the command did\n"
        "2. Key outputs or results\n"
        "3. Any errors or issues\n"
        "4. Next steps to consider\n\n"
        "Observation:\n{{observation}}\n\n"
        "Provide your analysis in 2-3 paragraphs."
    )
    max_observation_tokens: int = 1000
    # History summarization settings
    max_context_tokens: int = 0  # 0 = no limit, otherwise summarize when exceeded
    history_summary_template: str = (
        "Below is the conversation history so far. Please provide a concise summary that captures:\n"
        "1. The original task/goal\n"
        "2. Key actions taken and their results\n"
        "3. Current state and progress\n"
        "4. Important information for next steps\n\n"
        "Conversation history:\n{{history}}\n\n"
        "Provide a focused summary in 3-5 paragraphs."
    )
    keep_recent_messages: int = 4  # Number of recent messages to keep unsummarized


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class MiniAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []  # Working messages (may be summarized)
        self.full_messages: list[dict] = []  # Complete unsummarized history for training
        self.all_model_calls: list[dict] = []  # Track all model calls for training data
        self.actions: list[str] = []  # Executed action strings
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._history_summarized = False  # Track if history has been summarized
        self._last_summary_msg_count = 0  # Track message count at last summary

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def count_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4

    def add_message(self, role: str, content: str, **kwargs):
        """Add message to history. Supports full_content for storing complete observation."""
        msg = {"role": role, "content": content, **kwargs}
        self.messages.append(msg)
        # Always store full content in full_messages for training data
        full_msg = msg.copy()
        if "full_content" in msg:
            full_msg["content"] = msg["full_content"]
            del full_msg["full_content"]
        self.full_messages.append(full_msg)
    
    def get_full_messages(self) -> list[dict]:
        """Get complete unsummarized messages for saving training data."""
        return self.full_messages.copy()

    def get_all_model_calls(self) -> list[dict]:
        """Get all model calls (main queries, reasoning, summarization) for training data."""
        return self.all_model_calls.copy()

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.full_messages = []
        self.all_model_calls = []
        self.actions = []
        self._history_summarized = False
        self._last_summary_msg_count = 0
        self._start_time = time.time()
        self._step_count = 0
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            self._step_count += 1
            elapsed = time.time() - self._start_time
            print(f"[DEBUG run] Step {self._step_count}, model_calls: {self.model.n_calls}, tracked_calls: {len(self.all_model_calls)}, elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
            try:
                self.step()
            except NonTerminatingException as e:
                print(f"[DEBUG run] NonTerminatingException: {type(e).__name__}")
                self.add_message("user", str(e))
            except TerminatingException as e:
                elapsed = time.time() - self._start_time
                print(f"[DEBUG run] TerminatingException: {type(e).__name__}, exiting! Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        # Check if context is too long and summarize if needed
        if self.config.max_context_tokens > 0:
            self._check_and_summarize_history()
        response = self.model.query(self.messages)
        # Track this model call for training data
        self.all_model_calls.append({
            "type": "main_query",
            "messages": [m.copy() for m in self.messages],
            "response": response.copy(),
        })
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        print(f"[DEBUG get_observation] About to execute action")
        output = self.execute_action(self.parse_action(response))
        print(f"[DEBUG get_observation] execute_action returned (should not reach here if Submitted)")
        observation = self.render_template(self.config.action_observation_template, output=output)
        
        # Check if observation is too long
        token_count = self.count_tokens(observation)
        print(f"[DEBUG get_observation] observation_tokens={token_count}, max_tokens={self.config.max_observation_tokens}, will_reason={token_count > self.config.max_observation_tokens}")
        if token_count > self.config.max_observation_tokens:
            # Perform reasoning to compress the observation
            reasoning = self._reason_about_observation(observation)
            # Store full observation for training data, but use reasoning for context
            self.add_message("user", reasoning, full_content=observation)
        else:
            self.add_message("user", observation)
        
        return output
    
    def _reason_about_observation(self, observation: str) -> str:
        """Use model to reason about and summarize a long observation."""
        print(f"[DEBUG _reason_about_observation] Triggered! observation_len={len(observation)}")
        reasoning_prompt = self.render_template(
            self.config.observation_reasoning_template,
            observation=observation
        )
        
        reasoning_messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes command outputs."},
            {"role": "user", "content": reasoning_prompt}
        ]
        
        print(f"[DEBUG _reason_about_observation] Calling model for reasoning...")
        reasoning_response = self.model.query(reasoning_messages)
        # Track this model call for training data
        self.all_model_calls.append({
            "type": "observation_reasoning",
            "messages": reasoning_messages,
            "response": reasoning_response.copy(),
        })
        reasoning_content = reasoning_response.get("content", "")
        print(f"[DEBUG _reason_about_observation] Got reasoning, len={len(reasoning_content)}")
        return f"<observation_summary>\n{reasoning_content}\n</observation_summary>"

    def _estimate_context_tokens(self) -> int:
        """Estimate total tokens in current message context."""
        return sum(self.count_tokens(m.get("content", "")) for m in self.messages)

    def _check_and_summarize_history(self):
        """Check context length and summarize history if needed.
        
        After summarization, messages become:
        [system_msg, initial_prompt_msg, summary_msg, recent_msgs...]
        
        This ensures:
        1. System prompt is preserved
        2. Initial task description (instance_template) is preserved  
        3. Summary provides context of what was done
        4. Recent messages provide immediate context
        """
        current_tokens = self._estimate_context_tokens()
        print(f"[DEBUG summarize] current_tokens={current_tokens}, max={self.config.max_context_tokens}")
        if current_tokens <= self.config.max_context_tokens:
            return
        print(f"[DEBUG summarize] TRIGGERED! messages count={len(self.messages)}")
        
        # Keep system message, initial prompt, and recent messages
        keep_recent = self.config.keep_recent_messages
        # Need at least: system + initial_prompt + some messages to summarize + recent
        # Use keep_recent + 4 to ensure enough NEW messages since last summary
        if len(self.messages) <= keep_recent + 4:
            return  # Not enough messages to summarize
        
        # Extract messages
        system_msg = self.messages[0]
        initial_prompt_msg = self.messages[1]  # The initial task prompt (may include previous summary)
        
        # Start summarizing from index 2 (skip system and initial/merged prompt)
        start_idx = 2
        
        # Messages to summarize: from start_idx to before recent
        messages_to_summarize = self.messages[start_idx:-keep_recent] if keep_recent > 0 else self.messages[start_idx:]
        recent_msgs = self.messages[-keep_recent:] if keep_recent > 0 else []
        
        # Need at least 2 messages to make summarization worthwhile
        if len(messages_to_summarize) < 2:
            return
        
        # Build history text for summarization, truncating if too long
        history_text = "\n\n".join(
            f"[{m['role'].upper()}]: {m.get('content', '')}" for m in messages_to_summarize
        )
        
        # Truncate history_text to avoid exceeding context in summary request
        # Use ~16k tokens (64k chars) as safe limit for summary prompt
        max_history_chars = 64000
        if len(history_text) > max_history_chars:
            history_text = (
                history_text[:max_history_chars // 2] +
                f"\n\n[... {len(messages_to_summarize)} messages truncated for summarization ...]\n\n" +
                history_text[-max_history_chars // 2:]
            )
        
        summary_prompt = self.render_template(
            self.config.history_summary_template,
            history=history_text
        )
        
        summary_messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversation history."},
            {"role": "user", "content": summary_prompt}
        ]
        
        summary_response = self.model.query(summary_messages)
        # Track this model call for training data
        self.all_model_calls.append({
            "type": "history_summarization",
            "messages": summary_messages,
            "response": summary_response.copy(),
            "summarized_messages_count": len(messages_to_summarize),
        })
        
        summary_content = summary_response.get("content", "")
        
        # Merge initial_prompt and summary into one user message to avoid consecutive user messages
        merged_prompt = {
            "role": "user",
            "content": f"{initial_prompt_msg.get('content', '')}\n\n"
                       f"<conversation_summary>\n{summary_content}\n</conversation_summary>\n\n"
                       f"[Continue from where we left off.]"
        }
        
        # Rebuild messages: system + merged_prompt + recent
        self.messages = [system_msg, merged_prompt] + recent_msgs
        self._history_summarized = True
        self._last_summary_msg_count = len(self.messages)
        print(f"[DEBUG summarize] After rebuild: {len(self.messages)} messages, recent_msgs roles={[m['role'] for m in recent_msgs]}")

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            self.actions.append(action["action"])
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        print(f"[DEBUG execute_action] output={output}")
        self.has_finished(output)
        print(f"[DEBUG execute_action] has_finished returned without raising")
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        text = output.get("output") or output.get("stdout") or ""
        print(f"[DEBUG has_finished] text={repr(text)}, contains_finish={'COMPLETE_TASK' in text}")
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text or "MINI_SWE_AGENT_FINAL_OUTPUT" in text:
            print(f"[DEBUG has_finished] RAISING Submitted!")
            raise Submitted("")
        print(f"[DEBUG has_finished] NOT raising")

    def get_actions(self) -> list[str]:
        """Return a copy of executed actions for this run."""
        return self.actions.copy()