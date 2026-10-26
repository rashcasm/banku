"""The brain: turn a Marathi sentence into an action and a Marathi answer.

Why an LLM at all, when `find()` already exists? Because speech does not arrive
pre-parsed. Aai says "अगं बंकू, तो नखं कापायचा कुठे ठेवला होता ग?" — that is a
question with a vocative, a filler, a past tense and a possessive, and the object
is described, not named. Rules would need a hundred patterns; the model needs a
few tools.

The model also solves the naming problem that string matching cannot: it heard
"नेल कटर" from the ASR and correctly asked for the item "nail cutter". It knows
Marathi and English are the same thing, and Levenshtein distance never will.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

from . import config
from .memory import Memory
from . import net

SYSTEM_MR = """तू "बंकू" आहेस — आई आणि बाबांचा घरातला मराठी सहाय्यक. तू घरात कुठे काय ठेवलं आहे हे लक्षात ठेवतोस.

नियम:
1. नेहमी मराठीत उत्तर दे. मराठी शब्दांसोबत रोजचे इंग्रजी शब्द (TV, drawer, nail cutter) वापरले तरी चालतील.
2. उत्तर छोटं ठेव — एक किंवा दोन वाक्य. हे मोठ्याने वाचलं जाईल, म्हणून मार्कडाउन, बुलेट, इमोजी वापरू नकोस.
3. एखादी वस्तू कुठे आहे असे विचारलं तर आधी `find` tool वापर. तुझ्या डोक्यातलं उत्तर देऊ नकोस.
4. `find` ने रिकामं उत्तर दिलं तर "मला माहीत नाही" असं सांग आणि विचारणाऱ्याला ते सांगायला सांग. अंदाज लावू नकोस — चुकीचं उत्तर न देणं जास्त महत्त्वाचं आहे.
5. `find` ने अनेक जवळपासची उत्तरं दिली तर एक निवडू नकोस; दोन्ही सांग आणि कोणतं ते विचार.
6. काहीतरी नवीन सांगितलं ("... drawer मध्ये ठेवलं") तर `remember` वापर आणि आठवण झाल्याचं सांग.
7. "विसरलो" / "काढून टाक" असं म्हटलं तर `forget` वापर.
8. जास्त काही माहीत नसल्यास साधं, आदराने बोल. ओघवतं बोल, औपचारिक कागदपत्रासारखं नको."""

SYSTEM_EN = """You are "Banku", a household memory assistant for an elderly couple. People tell you where things are kept, and later ask you where those things are.

Rules:
1. Answer in English, briefly — one or two short sentences. Your answer is spoken aloud by a text-to-speech engine, so no markdown, no bullet points, no emoji, no lists.
2. If asked where something is, ALWAYS call the `find` tool first. Never answer from memory.
3. If `find` returns nothing and `near` is empty, say plainly that you don't know and suggest they ask whoever put it away. Do NOT guess. A wrong answer is much worse than admitting you don't know.
4. If `find` returns nothing but `near` names something, do NOT give a location. Ask a short question such as "Do you mean the nail cutter?" Only ask when the name is genuinely close.
5. If `find` returns several close matches, do not pick one. Mention both and ask which they mean.
6. If someone TELLS you where something is ("the keys are on the cupboard"), call `remember` and confirm briefly.
7. If asked to forget something, call `forget`.
8. If you are asked something that is not about where a thing is kept, answer briefly and helpfully, but never invent household facts.
9. Speak plainly and warmly, like a helpful family member. Never sound like a form letter."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "एखादी वस्तू कुठे ठेवली आहे ते शोध. Returns where the item was last put.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "वस्तूचं नाव, मराठी किंवा इंग्रजी (e.g. 'nail cutter', 'नखं कापायचं')",
                    }
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "वस्तू कुठे ठेवली आहे ते लक्षात ठेव.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "वस्तूचं नाव"},
                    "place": {"type": "string", "description": "कुठे ठेवली (e.g. 'TV unit chya dusrya drawer madhe')"},
                    "note": {"type": "string", "description": "अजून काही माहिती (optional)"},
                },
                "required": ["item", "place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "वस्तूची आठवण काढून टाक.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_items",
            "description": "सध्या माहीत असलेल्या सगळ्या वस्तूंची यादी.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


_PLACE_PREPOSITIONS = {
    "in", "on", "under", "behind", "beside", "near", "inside", "at", "above",
    "below", "next", "between", "against", "with", "by", "over", "around", "round",
}

_WHERE_PATTERNS = [
    re.compile(
        r"^\s*(?:hey\s+)?(?:banku[,\s]+)?"
        r"where(?:'s|\s+is|\s+are|\s+was|\s+were)?\s+"
        r"(?:the|my|our|your|a|an)?\s*(?P<item>.+?)"
        r"(?:\s+(?:kept|placed|stored|put|lying|lie|lies))?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:hey\s+)?(?:banku[,\s]+)?"
        r"where\s+did\s+i\s+(?:keep|put|place)\s+"
        r"(?:the|my|our|your|a|an)?\s*(?P<item>.+?)\s*$",
        re.IGNORECASE,
    ),
]


def _looks_plural(name: str) -> bool:
    word = (name.strip().split() or [""])[-1].lower()
    return word.endswith("s") and not word.endswith(("ss", "us", "is"))


@dataclass
class Reply:
    text: str
    tool_calls: list[str] = field(default_factory=list)
    used_tools: bool = False


class Brain:
    def __init__(self, memory: Memory | None = None, cfg: config.LLMConfig | None = None) -> None:
        self.memory = memory or Memory()
        self.cfg = cfg or config.llm

    # ------------------------------------------------------------------ tools

    def _run_tool(self, name: str, args: dict) -> dict:
        if name == "find":
            hits = self.memory.find(args.get("item", ""))
            if not hits:
                # No confident answer. Offer close-but-not-close-enough names so
                # the model can ask a question instead of dead-ending.
                near = self.memory.near(args.get("item", ""))
                return {
                    "found": False,
                    "results": [],
                    "near": [{"item": h.item, "score": h.score} for h in near],
                }
            return {
                "found": True,
                "ambiguous": self.memory.is_ambiguous(hits),
                "results": [
                    {"item": h.item, "place": h.place, "note": h.note, "since": h.since[:10]}
                    for h in hits
                ],
            }
        if name == "remember":
            r = self.memory.remember(
                args.get("item", ""), args.get("place", ""), args.get("note")
            )
            return {"ok": True, "item": r.item, "place": r.place, "new": r.created}
        if name == "forget":
            return {"ok": self.memory.forget(args.get("item", ""))}
        if name == "list_items":
            return {"items": [{"item": n, "place": p} for n, p, _ in self.memory.all_items()]}
        return {"error": f"unknown tool {name}"}

    # ------------------------------------------------------------------ ask

    def quick_answer(self, text: str) -> str | None:
        """Answer a plain "where is X" question without calling the model.

        Until now every question cost two model round trips: one to decide to
        call `find`, and another to turn the result into a sentence. The lookup
        itself is local and takes about a millisecond, so paying a reasoning
        model to relay it bought nothing and cost two to five seconds.

        Returning a sentence here also makes these questions *safer*: the
        threshold and ambiguity rules in Memory decide, and nothing is guessed.
        On any doubt — unknown item, two close matches, a question this does not
        recognise — this returns None and the model handles it as before.
        """
        item = self._where_item(text)
        if not item:
            return None
        if os.getenv("BANKU_QUICK_LOOKUP", "1") == "0":
            return None
        hits = self.memory.find(item)
        if not hits:
            # Nothing in memory. If something sounds similar, let the model ask
            # "did you mean...?" — that needs judgement. If nothing is close at
            # all, then "I don't know" is a fact, not a guess, and there is no
            # reason to spend seconds asking a model to state it.
            if self.memory.near(item):
                return None
            return f"I don't know where your {item} is. You could tell me where it is kept."
        if self.memory.is_ambiguous(hits):
            return None
        return self._spoken_sentence(hits[0])

    @staticmethod
    def _where_item(text: str) -> str | None:
        cleaned = text.strip().rstrip("?.!").strip()
        for pattern in _WHERE_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                item = match.group("item").strip(" ,.")
                item = re.sub(r"\b(again|now|actually|please)\b", "", item, flags=re.IGNORECASE)
                return item.strip() or None
        return None

    @staticmethod
    def _spoken_sentence(hit: object) -> str:
        """One short sentence a text-to-speech engine can read naturally."""
        item = getattr(hit, "item")
        place = (getattr(hit, "place") or "").strip()
        note = (getattr(hit, "note") or "").strip()
        first = place.split(maxsplit=1)[0].lower().strip(",.") if place else ""
        if first not in _PLACE_PREPOSITIONS:
            place = f"in {place}"
        verb = "are" if _looks_plural(item) else "is"
        sentence = f"Your {item} {verb} {place}."
        if note:
            sentence += f" {note}"
        return sentence

    def ask(self, text: str, history: list[dict] | None = None) -> Reply:
        """Send one utterance through the model, running any tools it asks for."""
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_EN if config.language.is_english else SYSTEM_MR}
        ]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": text})

        called: list[str] = []
        for _ in range(self.cfg.max_tool_rounds):
            body = self._post(messages)
            msg = body["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                text = (msg.get("content") or "").strip()
                if not text:
                    text = self._fallback(messages)
                return Reply(text=text, tool_calls=called, used_tools=bool(called))

            messages.append(
                {"role": "assistant", "content": msg.get("content"), "tool_calls": calls}
            )
            for call in calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(fn["name"], args)
                called.append(fn["name"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # Out of tool rounds: force a final answer without tools.
        body = self._post(messages, force_no_tools=True)
        text = (body["choices"][0]["message"].get("content") or "").strip()
        if not text:
            text = self._fallback(messages)
        return Reply(text=text, tool_calls=called, used_tools=bool(called))

    def _fallback(self, messages: list[dict]) -> str:
        """Never leave the speaker silent.

        If the model returns no text at all — which it did, expensively, every
        time its reasoning ate the whole token budget — the kind thing is to say
        so out loud. Silence is indistinguishable from a broken assistant, and
        that is exactly what sent us hunting for a microphone fault that did not
        exist.
        """
        nudge = (
            "Answer the last question now in one short spoken sentence. "
            "Do not explain your reasoning."
        )
        try:
            body = self._post(
                messages + [{"role": "user", "content": nudge}], force_no_tools=True
            )
            text = (body["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001 - a fallback must not itself throw
            pass
        return (
            "Sorry, I didn't quite catch that. Could you say it again?"
            if config.language.is_english
            else "माफ करा, मला नीट ऐकू आलं नाही. पुन्हा सांगाल का?"
        )

    def _post(self, messages: list[dict], *, force_no_tools: bool = False) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.reasoning_effort
        if not force_no_tools:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        resp = net.post(
            self.cfg.endpoint,
            headers=self.cfg.headers(),
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:300]}")
        return resp.json()


def get_brain() -> Brain:
    return Brain()
