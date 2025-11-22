"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
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
    step_limit: int = 0
    cost_limit: float = 3.0
    memory_template: str = (
        "REMINDER: Focus on completing the task described in the original instructions above. "
        "Follow the recommended workflow and stay on track with the requirements.\n\n"
        "# Memory (Summary of previous steps)\n{{memory}}\n\n"
        "# Last Action\n```bash\n{{last_action}}\n```\n\n"
        "# Last Observation\n{{last_observation}}\n\n"
        "# Analysis of Last Observation\n{{reasoning}}"
    )
    reasoning_template: str = (
        "Analyze the following command output and provide a concise summary (2-3 sentences) focusing on:\n"
        "1. What was found or what happened\n"
        "2. Any errors or important information\n"
        "3. What this tells us about the task progress\n\n"
        "Command: {{action}}\n\nOutput:\n{{observation}}\n\n"
        "Provide only the analysis, no other text."
    )
    summarize_template: str = (
        "Summarize the following conversation history into key points (max 200 words):\n\n{{history}}\n\n"
        "Focus on: decisions made, files modified, errors encountered, current progress, and optimization iterations completed.\n"
        "Specifically note:\n"
        "- How many optimization iterations have been completed\n"
        "- What benchmark results showed (performance improvements or regressions)\n"
        "- Whether tests passed\n"
        "- What the next optimization should focus on based on benchmark feedback\n"
        "- If 2-3 iterations are done, note that agent should prepare to submit soon\n"
        "Provide only the summary, no other text."
    )
    max_observation_for_reasoning: int = 3000  # Trigger reasoning for observations longer than this
    memory_update_frequency: int = 3  # Update memory summary every N steps


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


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.memory: str = ""
        self.last_action: str = ""
        self.last_observation: str = ""
        self.last_reasoning: str = ""
        self.step_count: int = 0
        self.history: list[dict] = []

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def _build_messages(self) -> list[dict]:
        """Build message list for current step: system + task + memory + last step."""
        messages = [
            {"role": "system", "content": self.render_template(self.config.system_template)},
            {"role": "user", "content": self.render_template(self.config.instance_template)}
        ]
        if self.step_count > 0:
            step_warning = ""
            if self.config.step_limit > 0:
                remaining = self.config.step_limit - self.model.n_calls
                if remaining <= 40:
                    step_warning = f"\n**URGENT: Only {remaining} steps remaining! Complete current iteration and SUBMIT NOW or you will fail!** \n"
                elif remaining <= 60:
                    step_warning = f"\n**WARNING: Only {remaining} steps left. This should be your LAST optimization iteration. Test, benchmark, and submit!**\n"
                elif remaining <= 80:
                    step_warning = f"\n**NOTICE: {remaining} steps remaining. You have room for 1 more complete iteration (optimize→test→benchmark). Plan accordingly.**\n"
            
            memory_content = self.render_template(
                self.config.memory_template,
                memory=self.memory or "No previous steps yet.",
                last_action=self.last_action,
                last_observation=self.last_observation,
                reasoning=self.last_reasoning
            )
            messages.append({"role": "user", "content": step_warning + memory_content})
        return messages

    def _get_reasoning(self, action: str, observation: str) -> str:
        """Use LLM to analyze and reason about the observation."""
        if len(observation) < self.config.max_observation_for_reasoning:
            return observation
        reasoning_prompt = self.render_template(
            self.config.reasoning_template,
            action=action,
            observation=observation
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes command outputs."},
            {"role": "user", "content": reasoning_prompt}
        ]
        response = self.model.query(messages)
        return response["content"]

    def _update_memory(self):
        """Summarize history into memory using LLM."""
        if not self.history:
            return
        history_text = "\n\n".join([
            f"Step {i+1}:\nAction: {h['action']}\nObservation: {h['observation'][:500]}...\nReasoning: {h['reasoning']}"
            for i, h in enumerate(self.history)
        ])
        summary_prompt = self.render_template(self.config.summarize_template, history=history_text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversation history."},
            {"role": "user", "content": summary_prompt}
        ]
        response = self.model.query(messages)
        self.memory = response["content"]
        self.history = []

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.memory = ""
        self.last_action = ""
        self.last_observation = ""
        self.last_reasoning = ""
        self.step_count = 0
        self.history = []
        while True:
            try:
                self.step()
                self.step_count += 1
                if self.config.memory_update_frequency > 0 and self.step_count % self.config.memory_update_frequency == 0:
                    self._update_memory()
            except NonTerminatingException as e:
                self.last_observation = str(e)
                self.last_reasoning = "Error occurred, need to try a different approach."
            except TerminatingException as e:
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        messages = self._build_messages()
        return self.model.query(messages)

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        action = self.parse_action(response)
        output = self.execute_action(action)
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.last_action = action["action"]
        self.last_observation = observation
        self.last_reasoning = self._get_reasoning(self.last_action, observation)
        self.history.append({
            "action": self.last_action,
            "observation": observation,
            "reasoning": self.last_reasoning
        })
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
