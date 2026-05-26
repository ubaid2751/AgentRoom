from __future__ import annotations

import json
import asyncio
from typing import TYPE_CHECKING

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent.base import AgentAction, CognitiveAgent

if TYPE_CHECKING:
    from app.environment.state import EnvironmentSnapshot

APP_NAME = "rmcs_simulation"


class SessionManager:
    """
    Owns one private ADK session per agent and one global conversation session.

    Private session  — full ADK conversation history for that agent.
                       Contains their role context, memory, reasoning across rounds.
                       No other agent can read this.

    Global session   — append-only public speech log.
                       Only `speech` from AgentAction is written here.
                       This is what every agent reads at the start of each round.
    """

    def __init__(self):
        self.service = InMemorySessionService()
        self._runners:          dict[str, Runner]        = {}
        self._agents:           dict[str, CognitiveAgent] = {}

    async def register_agent(self, agent: CognitiveAgent):
        """
        Create a private LlmAgent + Runner + Session for this cognitive agent.
        The system prompt (which contains the secret role) is baked in here.
        """
        llm_agent = LlmAgent(
            name=agent.agent_id,
            model=agent.model,
            instruction=agent.system_prompt,
        )

        runner = Runner(
            agent=llm_agent,
            app_name=APP_NAME,
            session_service=self.service,
        )

        await self.service.create_session(
            app_name=APP_NAME,
            user_id=agent.agent_id,
            session_id=f"private_{agent.agent_id}",
        )

        self._runners[agent.agent_id] = runner
        self._agents[agent.agent_id]  = agent

        print(f"  [SessionManager] Registered {agent.agent_id} ({agent.role.value})")


    async def think(
        self,
        agent_id: str,
        snapshot: "EnvironmentSnapshot",
    ) -> AgentAction:
        """
        Inject the snapshot as a user message into the agent's private session.
        The agent reasons from their full private history + current snapshot.
        Returns a structured AgentAction.
        """
        agent   = self._agents[agent_id]
        runner  = self._runners[agent_id]

        turn_prompt = agent.build_turn_prompt(snapshot)

        message = types.Content(
            role="user",
            parts=[types.Part(text=turn_prompt)],
        )

        raw_response = ""
        async for event in runner.run_async(
            user_id=agent_id,
            session_id=f"private_{agent_id}",
            new_message=message,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    raw_response = event.content.parts[0].text
                break

        action = self._parse_action(raw_response, agent_id)

        agent.memory.update_from_action(action)
        agent.memory.add_reflection(
            f"Round {snapshot.round}: I thought — {action.inner_thought[:80]}..."
        )

        return action


    def _parse_action(self, raw: str, agent_id: str) -> AgentAction:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return AgentAction(**json.loads(cleaned.strip()))
        except Exception as e:
            print(f"  [SessionManager] Parse error for {agent_id}: {e}")
            return AgentAction(
                inner_thought="(parse error — staying quiet)",
                speech="I need a moment to think.",
                suspicion={},
            )
