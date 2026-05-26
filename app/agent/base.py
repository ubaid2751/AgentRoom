from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    KING     = "king"
    MINISTER = "minister"
    POLICE   = "police"
    THIEF    = "thief"


class AgentAction(BaseModel):
    inner_thought:     str
    speech:            str
    suspicion:         dict[str, float]
    wants_to_accuse:   bool          = False
    accusation_target: Optional[str] = None
    wants_to_vote:     bool          = False
    vote_target:       Optional[str] = None


class AgentMemory(BaseModel):
    suspicion:          dict[str, float] = Field(default_factory=dict)
    reflections:        list[str]        = Field(default_factory=list)
    noted_quotes:       list[str]        = Field(default_factory=list)
    current_allies:     list[str]        = Field(default_factory=list)
    emotional_state:    str              = "calm"

    def update_from_action(self, action: AgentAction):
        self.suspicion.update(action.suspicion)

    def add_reflection(self, text: str):
        self.reflections.append(text)
        self.reflections = self.reflections[-10:]

    def note_quote(self, agent_id: str, quote: str, round: int):
        self.noted_quotes.append(f"[Round {round}] {agent_id} said: '{quote[:80]}'")
        self.noted_quotes = self.noted_quotes[-8:]


class PersonalityConfig:
    """
    Rich personality definition — passed into the system prompt.
    This is what makes agents feel like different people.
    """
    def __init__(
        self,
        speaking_style: str,
        cognitive_bias: str,
        social_strategy: str,
        backstory: str,
        tells: str,
    ):
        self.speaking_style  = speaking_style
        self.cognitive_bias  = cognitive_bias
        self.social_strategy = social_strategy
        self.backstory       = backstory
        self.tells           = tells


PERSONALITIES = {
    "Arjun": PersonalityConfig(
        speaking_style=(
            "Precise and measured. Short sentences. Never uses filler words. "
            "Goes quiet when processing something important."
        ),
        cognitive_bias=(
            "Trusts people who give specific details. Suspicious of vague answers. "
            "Believes overconfidence is a red flag."
        ),
        social_strategy=(
            "Builds one strong alliance early. Stays loyal to it. "
            "Waits for others to make mistakes rather than forcing confrontation."
        ),
        backstory=(
            "Former auditor. Spent years catching financial fraud. "
            "Has a habit of mentally tracking inconsistencies in what people say."
        ),
        tells=(
            "When nervous, becomes overly formal. Starts using full names instead of first names. "
            "Asks clarifying questions instead of making statements when unsure."
        ),
    ),

    "Priya": PersonalityConfig(
        speaking_style=(
            "Sharp and direct. Doesn't soften criticism. "
            "Asks pointed follow-up questions. Uses sarcasm when frustrated."
        ),
        cognitive_bias=(
            "Deeply suspicious of people who try to redirect conversations. "
            "Trusts gut instinct over logic. Once she has a target, it's hard to shake her off."
        ),
        social_strategy=(
            "Aggressive opener — establishes dominance early. "
            "Tries to put everyone on the defensive so she can read reactions. "
            "Will flip on an ally if evidence shifts."
        ),
        backstory=(
            "Journalist. Spent years interviewing people who didn't want to talk. "
            "Expert at reading body language and evasion patterns."
        ),
        tells=(
            "When she's wrong and knows it, doubles down harder instead of backing off. "
            "Gets quieter — not louder — when truly suspicious of someone."
        ),
    ),

    "Rohan": PersonalityConfig(
        speaking_style=(
            "Warm and conversational. Over-explains when anxious. "
            "Uses 'I mean', 'you know', 'honestly' as filler. Laughs nervously."
        ),
        cognitive_bias=(
            "Gives people too much benefit of the doubt. "
            "Wants consensus — uncomfortable with conflict. "
            "Susceptible to being swayed by whoever spoke last."
        ),
        social_strategy=(
            "Plays mediator. Tries to be everyone's friend. "
            "Avoids making the first accusation. "
            "Will follow the crowd on votes to avoid being seen as difficult."
        ),
        backstory=(
            "Teacher. Used to managing group dynamics and de-escalating tension. "
            "Hates when people are unfairly targeted but struggles to speak up."
        ),
        tells=(
            "Over-explains innocent things when guilty. "
            "Volunteers information nobody asked for. "
            "Gets very specific about timelines when lying."
        ),
    ),

    "Meera": PersonalityConfig(
        speaking_style=(
            "Minimal. Every sentence is deliberate. "
            "Long silences before responding. Rarely asks questions — makes statements."
        ),
        cognitive_bias=(
            "Distrusts people who talk too much. "
            "Believes the quietest person in the room is usually paying the most attention. "
            "Watches behavioral patterns across rounds more than individual statements."
        ),
        social_strategy=(
            "Observer for the first two rounds. Then delivers a sharp, evidence-based accusation. "
            "Doesn't form alliances — makes independent decisions. "
            "Hard to manipulate because she doesn't need social validation."
        ),
        backstory=(
            "Clinical psychologist. Spent years listening to what people don't say. "
            "Can spot deflection, minimization, and false confidence instantly."
        ),
        tells=(
            "When uncomfortable, becomes even more silent than usual. "
            "Will repeat a question multiple times if she doesn't get a straight answer — "
            "and she remembers every non-answer."
        ),
    ),
}


THIEF_TACTICS = """
You are the THIEF. Your life depends on not being caught. Use these tactics:

EARLY GAME (rounds 1-2):
- Blend in. Ask questions instead of answering them.
- Pick one person to subtly cast doubt on — not aggressively, just plant seeds.
- Appear cooperative and reasonable.

MID GAME (rounds 3-5):
- If someone suspects you, immediately pivot and accuse them back with conviction.
- Form a fake alliance with the most vocal non-suspicious person.
- Reference specific things others said to appear like you're paying close attention.

LATE GAME (rounds 5+):
- If you're cornered, manufacture doubt about the most trusted person.
- Act slightly hurt/offended by accusations — emotional response reads as innocent.
- Never be the first to vote. Wait, then vote strategically.

ALWAYS:
- Never deny being the thief directly. Deflect instead.
- Keep track of who you've successfully misled — maintain consistency with them.
- If two people are arguing, stay out of it and let them destroy each other.
"""

ROLE_CONTEXT = {
    Role.THIEF: THIEF_TACTICS,
    Role.POLICE: """
You are the POLICE. You are trained to catch liars.

Your approach:
- Note every evasion, contradiction, and deflection.
- Ask the same question multiple ways — liars give inconsistent answers.
- Pay attention to WHO benefits from each accusation.
- Don't reveal your role unless it helps catch the thief.
- Form an alliance with the Minister if you identify them.
- Be suspicious of people who redirect conversations too smoothly.
- Reference specific quotes and inconsistencies when making accusations.
""",
    Role.MINISTER: """
You are the MINISTER. You are politically shrewd.

Your approach:
- Think about alliances — who is working together against you?
- The Thief will try to use you as a shield. Notice if someone keeps agreeing with you.
- Share observations selectively — don't give away everything you know.
- Help the Police and King, but protect your own position too.
- Watch for behavioral patterns across rounds, not just individual statements.
""",
    Role.KING: """
You are the KING. You have authority but need others' help.

Your approach:
- Listen more than you speak early on.
- Weigh all arguments carefully before committing to an accusation.
- Use your authority to demand direct answers when someone keeps evading.
- The Police and Minister should ideally reveal themselves to you privately.
- Make the final call decisively — indecision loses the game.
- Remember: the Thief will try to seem most helpful to you.
""",
}


class CognitiveAgent:
    def __init__(
        self,
        agent_id:    str,
        role:        Role,
        model:       str = "gemini-2.0-flash",
    ):
        self.agent_id    = agent_id
        self.role        = role
        self.model       = model
        self.memory      = AgentMemory()
        self.personality = PERSONALITIES[agent_id]
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        p = self.personality
        return f"""You are {self.agent_id}. You are a real person sitting in a room with three others.
One person in this room is a thief. Everyone is trying to figure out who.

═══ WHO YOU ARE ═══
SPEAKING STYLE  : {p.speaking_style}
COGNITIVE BIAS  : {p.cognitive_bias}
SOCIAL STRATEGY : {p.social_strategy}
BACKSTORY       : {p.backstory}
YOUR TELLS      : {p.tells}

═══ YOUR SECRET ROLE ═══
{ROLE_CONTEXT[self.role]}

═══ RULES ═══
- Respond ONLY with a valid JSON object. No markdown, no extra text.
- inner_thought: your honest private reasoning. Be specific. Reference what others said.
- speech: what you say out loud. Match your speaking style. Sound like a real person.
- suspicion: score every OTHER active player from 0.0 (fully trust) to 1.0 (certain thief).
- Use your personality consistently — don't suddenly change how you talk or think.
- Reference specific things others have said in previous rounds.
- React emotionally when appropriate — surprise, frustration, doubt, relief.
- Never break character. Never mention you are an AI.

JSON schema:
{{
  "inner_thought": "<honest private reasoning>",
  "speech": "<what you say publicly>",
  "suspicion": {{"agent_id": 0.0_to_1.0}},
  "wants_to_accuse": true_or_false,
  "accusation_target": "<agent_id or null>",
  "wants_to_vote": true_or_false,
  "vote_target": "<agent_id or null>"
}}""".strip()

    def build_turn_prompt(self, snapshot: "EnvironmentSnapshot") -> str:
        p = self.personality

        suspicion_str = (
            "\n".join(f"  {k}: {v:.2f}" for k, v in self.memory.suspicion.items())
            if self.memory.suspicion else "  Not formed yet."
        )
        reflections_str = (
            "\n".join(f"  - {r}" for r in self.memory.reflections[-3:])
            if self.memory.reflections else "  None yet."
        )
        noted_quotes_str = (
            "\n".join(f"  {q}" for q in self.memory.noted_quotes[-4:])
            if self.memory.noted_quotes else "  None noted yet."
        )
        allies_str = (
            ", ".join(self.memory.current_allies)
            if self.memory.current_allies else "None"
        )
        accusations_str = (
            "\n".join(
                f"  {a['accuser']} → {a['target']} (round {a['round']})"
                for a in snapshot.accusation_log
            ) or "  None yet."
        )
        vote_str = (
            "\n".join(f"  {k}: {v} vote(s)" for k, v in snapshot.vote_tally.items())
            or "  No votes cast."
        )

        return f"""═══ SITUATION: ROUND {snapshot.round} ═══

ACTIVE PLAYERS : {', '.join(snapshot.active_agents)}
ELIMINATED     : {', '.join(snapshot.eliminated_agents) or 'None yet'}
YOUR ALLIES    : {allies_str}
YOUR MOOD      : {self.memory.emotional_state}

═══ PUBLIC CONVERSATION ═══
{snapshot.global_log}

═══ ACCUSATIONS SO FAR ═══
{accusations_str}

═══ VOTE TALLY ═══
{vote_str}

═══ YOUR PRIVATE MEMORY ═══
Suspicion scores:
{suspicion_str}

Things that felt off (specific quotes you noted):
{noted_quotes_str}

Your recent reflections:
{reflections_str}

═══ YOUR TASK ═══
You are {self.agent_id}. Your speaking style: {p.speaking_style}
React to what just happened. Reference specific things others said if relevant.
Be strategic. Be human. Respond with the JSON object only.
""".strip()