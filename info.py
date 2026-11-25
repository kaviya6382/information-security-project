"""Rock Paper Scissors player.

Idea: every opponent bot is a deterministic function of *my* past plays (or of
the clock), so I run a small simulation of each candidate behaviour, keep score
of which one has been predicting the opponent correctly lately, and play the
move that beats that model's prediction.
"""
import random
from collections import deque

BEAT = {"R": "P", "P": "S", "S": "R"}   # BEAT[x] is the move that beats x
WINDOW = 20                              # how many recent rounds count for scoring
MIN_RATE = 0.5                           # below this, nobody is predictable -> random


class _AbbeyModel:
    """Mirror of a bot that counts my 2-move sequences and counters my likely next move."""

    def __init__(self):
        self.hist = []
        self.order = {a + b: 0 for a in "RPS" for b in "RPS"}

    def next_move(self, my_last):
        prev = my_last or "R"
        self.hist.append(prev)
        last_two = "".join(self.hist[-2:])
        if len(last_two) == 2:
            self.order[last_two] += 1
        sub = {k: self.order[k] for k in (prev + "R", prev + "P", prev + "S")}
        predicted_me = max(sub, key=sub.get)[-1:]
        return BEAT[predicted_me]


class _Match:
    def __init__(self, carried_abbey=None):
        self.me = []                 # my plays so far
        self.opp = []                # opponent plays so far
        self.preds = {}              # model name -> its prediction for the pending round
        self.ctx_seen = None         # my last two plays when the pending round was predicted
        self.ctx_counts = {}         # (my last two) -> {opp move: count}
        self.freq = {"R": 0, "P": 0, "S": 0}
        self.abbey = _AbbeyModel()
        self.abbey_carry = carried_abbey or _AbbeyModel()   # abbey state can survive across matches
        self.hits = {}

    # ---- learning from the opponent's last move -------------------------
    def observe(self, opp_move):
        self.opp.append(opp_move)
        self.freq[opp_move] += 1
        for name, pred in self.preds.items():
            if pred is not None:
                self.hits.setdefault(name, deque(maxlen=WINDOW)).append(int(pred == opp_move))
        if self.ctx_seen is not None:
            slot = self.ctx_counts.setdefault(self.ctx_seen, {"R": 0, "P": 0, "S": 0})
            slot[opp_move] += 1

    # ---- predictions of what the opponent will play next ----------------
    def predict_all(self):
        last = self.me[-1] if self.me else ""
        ctx = tuple(self.me[-2:])
        self.ctx_seen = ctx

        # kris: plays whatever beats my previous move
        kris = BEAT[last or "R"]

        # mrugesh: counters my most frequent move among my last 10 (history starts with "")
        last_ten = ([""] + self.me)[-10:]
        most = max(set(last_ten), key=last_ten.count)
        mrugesh = BEAT["S" if most == "" else most]

        # abbey: counts my 2-move sequences
        abbey = self.abbey.next_move(last)
        abbey_carry = self.abbey_carry.next_move(last)

        # quincy: repeats a cycle of length 5, so it equals what it did 5 rounds ago
        cycle5 = self.opp[-5] if len(self.opp) >= 5 else None

        # generic fallbacks
        slot = self.ctx_counts.get(ctx)
        ctx_pred = max(slot, key=slot.get) if slot and max(slot.values()) > 0 else None
        freq = max(self.freq, key=self.freq.get) if self.opp else None

        self.preds = {
            "cycle5": cycle5, "kris": kris, "mrugesh": mrugesh,
            "abbey": abbey, "abbey_carry": abbey_carry,
            "ctx": ctx_pred, "freq": freq,
        }

    def rate(self, name):
        h = self.hits.get(name)
        return (sum(h) + 1) / (len(h) + 2) if h else 0.5

    def best_model(self):
        return max(self.preds, key=self.rate)


_match = None


def player(prev_play, opponent_history=[]):
    global _match

    if prev_play == "" or _match is None:          # first game of a match -> reset
        carried = None
        if _match is not None:                     # an abbey-like bot keeps its counts between matches
            for name in ("abbey", "abbey_carry"):
                if _match.rate(name) > 0.9:
                    carried = getattr(_match, name)
                    break
        _match = _Match(carried)

    if prev_play:
        opponent_history.append(prev_play)
        _match.observe(prev_play)

    _match.predict_all()
    best = _match.best_model()
    pred = _match.preds[best]

    if pred is None or _match.rate(best) < MIN_RATE:
        move = random.choice("RPS")
    else:
        move = BEAT[pred]

    _match.me.append(move)
    return move
