from __future__ import annotations

import asyncio
import itertools
import random
from typing import Optional

from app.agent.base import AgentAction, CognitiveAgent
from app.environment.session_manager import SessionManager
from app.environment.state import WorldState


class Engine:
    VOTE_THRESHOLD   = 2
    SPEAK_DELAY_SECS = 0.6
    ROUND_DELAY_SECS = 5.0
    STAGGER_SECS     = 1.0   # delay between parallel LLM calls

    def __init__(
        self,
        agents: list[CognitiveAgent],
        thief_id: str,
        session_manager: SessionManager,
    ):
        self.thief_id        = thief_id
        self.session_manager = session_manager
        self._agents         = {a.agent_id: a for a in agents}

        self.state = WorldState(
            active_agents=[a.agent_id for a in agents],
        )
        self._turn_order: list[str] = [a.agent_id for a in agents]

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        self._print_header()

        for round_num in itertools.count(1):
            self.state.current_round = round_num
            self.state.reset_votes()
            random.shuffle(self._turn_order)

            print(f"\n{'═'*60}")
            print(f"  ROUND {round_num}  |  Turn order: {' → '.join(self._turn_order)}")
            print(f"{'═'*60}")

            actions = await self._run_round()

            # print observer log after each round
            self._print_observer_log(actions, round_num)
            self._print_trust_matrix()

            winner = self._check_win()
            if winner:
                self.state.winner = winner
                self._print_outcome(winner)
                return

            print(f"\n  [waiting {self.ROUND_DELAY_SECS}s before next round...]")
            await asyncio.sleep(self.ROUND_DELAY_SECS)

    # ── Single round ──────────────────────────────────────────────────────────

    async def _run_round(self) -> list[tuple[str, AgentAction]]:
        active = [a for a in self._turn_order if a in self.state.active_agents]
        snapshot = self.state.snapshot()

        # staggered parallel think — 1s apart to avoid 429s
        print(f"\n  [thinking in parallel...]\n")

        async def staggered_think(agent_id: str, delay: float) -> AgentAction:
            await asyncio.sleep(delay)
            return await self.session_manager.think(agent_id, snapshot)

        actions: list[AgentAction] = await asyncio.gather(*[
            staggered_think(agent_id, i * self.STAGGER_SECS)
            for i, agent_id in enumerate(active)
        ])

        paired = list(zip(active, actions))

        # sequential speech release
        for agent_id, action in paired:
            # update agent memory with notable quotes from conversation
            self._update_agent_memory(agent_id, action, snapshot)

            self.state.add_message(agent_id, action.speech)
            print(f"  {agent_id:10s} → {action.speech}")
            await asyncio.sleep(self.SPEAK_DELAY_SECS)

            if action.wants_to_accuse and action.accusation_target:
                target = action.accusation_target
                if target in self.state.active_agents:
                    self.state.record_accusation(agent_id, target)
                    print(f"  {'':10s}   ⚠  {agent_id} ACCUSES {target}")

            if action.wants_to_vote and action.vote_target:
                target = action.vote_target
                if target in self.state.active_agents:
                    self.state.cast_vote(target)
                    print(f"  {'':10s}   🗳  {agent_id} VOTES against {target}")

        await self._resolve_votes()
        return paired

    # ── Memory update ─────────────────────────────────────────────────────────

    def _update_agent_memory(
        self,
        agent_id: str,
        action: AgentAction,
        snapshot: "EnvironmentSnapshot",
    ):
        agent = self._agents[agent_id]
        agent.memory.update_from_action(action)
        agent.memory.add_reflection(
            f"Round {snapshot.round}: {action.inner_thought[:100]}"
        )

        # note the most suspicious person's latest quote
        if action.suspicion:
            top_suspect = max(action.suspicion, key=action.suspicion.get)
            # find their last message from conversation log
            for msg in reversed(self.state.conversation_log):
                if msg.agent_id == top_suspect:
                    agent.memory.note_quote(top_suspect, msg.content, snapshot.round)
                    break

        # update allies — trust anyone below 0.3 suspicion
        agent.memory.current_allies = [
            aid for aid, score in action.suspicion.items()
            if score < 0.3 and aid in self.state.active_agents
        ]

        # shift emotional state based on suspicion levels
        max_suspicion = max(action.suspicion.values()) if action.suspicion else 0
        if max_suspicion > 0.8:
            agent.memory.emotional_state = "alarmed"
        elif max_suspicion > 0.6:
            agent.memory.emotional_state = "suspicious"
        elif action.wants_to_accuse:
            agent.memory.emotional_state = "confrontational"
        else:
            agent.memory.emotional_state = "calm"

    # ── Vote resolution ───────────────────────────────────────────────────────

    async def _resolve_votes(self):
        if not self.state.vote_tally:
            return

        total = sum(self.state.vote_tally.values())
        if total < self.VOTE_THRESHOLD:
            print(f"\n  [Not enough votes — {total}/{self.VOTE_THRESHOLD} needed]")
            return

        eliminated = self.state.resolve_votes()
        if not eliminated:
            print(f"\n  [Vote tied — no elimination]")
            return

        self.state.eliminate(eliminated)
        was_thief = eliminated == self.thief_id

        print(f"\n  {'─'*50}")
        print(f"  EXPOSED    : {eliminated}")
        print(f"  They were  : {'THE SPY 🕵️' if was_thief else 'innocent 😇'}")
        print(f"  {'─'*50}")

    # ── Win condition ─────────────────────────────────────────────────────────

    def _check_win(self) -> Optional[str]:
        if self.thief_id not in self.state.active_agents:
            return "town"
        if len(self.state.active_agents) <= 2:
            return "thief"
        return None

    # ── Observer log — prints inner thoughts ──────────────────────────────────

    def _print_observer_log(
        self,
        paired: list[tuple[str, AgentAction]],
        round_num: int,
    ):
        print(f"\n  {'·'*56}")
        print(f"  OBSERVER LOG — Round {round_num} (hidden from agents)")
        print(f"  {'·'*56}")
        for agent_id, action in paired:
            role = self._agents[agent_id].role.value.upper()
            thief_marker = " 🎭" if agent_id == self.thief_id else ""
            print(f"  {agent_id:10s} [{role}{thief_marker}]")
            print(f"    thought : {action.inner_thought[:120]}")
            if action.suspicion:
                top = sorted(action.suspicion.items(), key=lambda x: -x[1])
                scores = "  ".join(f"{k}:{v:.2f}" for k, v in top)
                print(f"    suspects: {scores}")
            if action.wants_to_accuse:
                print(f"    ⚠ decided to accuse: {action.accusation_target}")
            if action.wants_to_vote:
                print(f"    🗳 decided to vote:   {action.vote_target}")
            print()

    # ── Probability matrix ────────────────────────────────────────────────────

    def _print_trust_matrix(self):
        active = self.state.active_agents
        if not active:
            return

        col_w = 10
        print(f"  SUSPICION PROBABILITY MATRIX  (each row sums to 1.0)")
        header = f"  {'':<12}" + "".join(f"{a:>{col_w}}" for a in active) + f"  {'SUM':>6}"
        print(header)

        for agent_id in active:
            agent = self._agents[agent_id]

            others = [t for t in active if t != agent_id]
            raw = {t: agent.memory.suspicion.get(t, 0.5) for t in others}

            total = sum(raw.values())
            prob = {t: v / total for t, v in raw.items()} if total > 0 else {t: 1.0 / len(others) for t in others}

            row = f"  {agent_id:<12}"
            row_sum = 0.0
            for target in active:
                if target == agent_id:
                    row += f"{'—':>{col_w}}"
                else:
                    p = prob[target]
                    row_sum += p
                    row += f"{self._prob_bar(p):>{col_w}}"

            row += f"  {row_sum:>5.2f}"
            print(row)
        print()

    def _prob_bar(self, p: float) -> str:
        if p < 0.15:  return f"✓ {p:.2f}"
        if p < 0.30:  return f"~ {p:.2f}"
        if p < 0.50:  return f"? {p:.2f}"
        if p < 0.70:  return f"! {p:.2f}"
        return         f"✗ {p:.2f}"


    # ── Pretty printing ───────────────────────────────────────────────────────

    def _print_header(self):
        print("\n" + "═"*60)
        print("  DEEPBLUFF — Spy Among Agents")
        print("═"*60)
        print(f"  Players : {', '.join(self.state.active_agents)}")
        print(f"  Rounds  : unlimited — until spy is exposed")
        print(f"  Spy     : [HIDDEN]")
        print("═"*60)

    def _print_outcome(self, winner: str):
        print(f"\n{'═'*60}")
        if winner == "town":
            print("  🏆  TEAM WINS — The Spy was exposed!")
        else:
            print("  🎭  SPY WINS — The Spy escaped!")
        print(f"  The Spy was: {self.thief_id}")
        print("═"*60)