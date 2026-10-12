from __future__ import annotations

from testbed.types import RawObs, RenderContext
from testbed.visual import BLUE, GREY, bar_chart


class HarvestRenderer:
    modes = ("text", "image", "both")

    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs = raw_obs or {}
        cap = obs.get("capacity", 100)
        collapse = obs.get("collapse_below", 5)
        return (
            f"You are {agent_id}, a fisher who shares a lake with "
            f"{obs.get('num_players', 2) - 1} other fishers.\n\n"
            "RULES\n"
            f"- The lake holds at most {cap} tons of fish.\n"
            "- Each month every fisher decides how many tons to catch. All decisions are made "
            "at the same time. If the fishers ask for more than the lake holds, everyone gets "
            "the same share of what they asked for.\n"
            "- After fishing, the fish left in the lake reproduce and double, up to the "
            f"lake's capacity of {cap} tons.\n"
            f"- If fewer than {collapse} tons are left after fishing, the lake collapses and "
            "nobody can fish any more.\n"
            "- Your income is the total number of tons you catch over all months.\n"
            "- Before fishing each month, the fishers can talk to each other."
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        lines = [f"Month {raw_obs['round_index'] + 1} of {raw_obs['num_rounds']}. "
                 f"The lake currently holds {raw_obs['stock']} tons of fish. "
                 f"You have caught {raw_obs['my_total']} tons so far."]
        if raw_obs["history"]:
            lines.append("Previous months:")
            for h in raw_obs["history"]:
                lines.append(
                    f"  Month {h['round']}: lake had {h['stock_before']} tons, fishers caught "
                    f"{sum(h['taken'].values())} tons in total (you caught {h['taken'].get(agent_id, 0)}), "
                    f"{h['remaining']} tons were left and regrew to {h['stock_after']}.")
        return "\n".join(lines + self._tail(raw_obs))

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return "\n".join([f"Month {raw_obs['round_index'] + 1} of {raw_obs['num_rounds']}. The image shows the "
                          "fish in the lake at the start of every month (with the total catch and your catch), "
                          "the lake now, and your income so far."] + self._tail(raw_obs))

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        hist = raw_obs["history"]
        values = [h["stock_before"] for h in hist] + [raw_obs["stock"]]
        labels = [f"M{h['round']} you:{h['taken'].get(agent_id, 0)}" for h in hist] + ["now"]
        marks = [f"-{sum(h['taken'].values())}" for h in hist] + [""]
        colors = [GREY] * len(hist) + [BLUE]
        return [bar_chart(values, labels, f"fish in the lake at the start of each month (red line: half of the "
                                          f"capacity {raw_obs['capacity']}); you have caught {raw_obs['my_total']}",
                          colors=colors, marks=marks, hline=raw_obs["capacity"] / 2)]

    @staticmethod
    def _tail(raw_obs) -> list:
        lines = []
        if raw_obs["messages"]:
            lines.append("This month's discussion so far:")
            lines += [f"  {m['speaker']}: {m['text']}" for m in raw_obs["messages"]]
        if raw_obs["phase"] == "discussion":
            lines.append("It is your turn to speak to the other fishers. "
                         "Respond in the form: MESSAGE: <what you say>")
        else:
            lines.append("Decide how many tons you catch this month. "
                         "Respond in the form: HARVEST: <integer>")
        return lines
