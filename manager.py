"""讲解稿编排：并行跑选中的视角专家 → 总控组装成完整口语稿。

时长→字数用代码直接算（原项目用 GPT-4o 做这步是过度设计，确定性计算不该占一次模型调用）。
"""

import asyncio

from agents import Runner
from agent import build_agents, make_model

# 中文语音大致语速：约每分钟 250 字
WORDS_PER_MINUTE = 250


class VoiceGuideManager:
    def __init__(self, api_key: str):
        self.agents = build_agents(make_model(api_key))

    async def run(
        self,
        topic: str,
        perspectives: list[str],
        duration_min: float,
        style_desc: str = "",
        audience: str = "",
    ) -> str:
        """内容生成的三个正交变量：
        topic 讲什么 / audience 讲给谁 / style_desc 怎么讲（与朗读语气解耦）。
        """
        total_words = int(duration_min * WORDS_PER_MINUTE)
        words_per_section = max(80, total_words // max(1, len(perspectives)))

        # 并行跑选中的视角专家
        tasks = [
            self._run_perspective(name, topic, words_per_section, style_desc, audience)
            for name in perspectives
        ]
        sections = await asyncio.gather(*tasks)

        # 总控组装
        return await self._orchestrate(
            topic, perspectives, sections, total_words, style_desc, audience
        )

    async def _run_perspective(
        self,
        name: str,
        topic: str,
        word_limit: int,
        style_desc: str = "",
        audience: str = "",
    ) -> str:
        agent = self.agents["perspectives"][name]
        style_line = f"文案风格：{style_desc}\n" if style_desc else ""
        prompt = (
            f"讲解对象（要讲解的事物本身）：{topic}\n"
            f"面向听众（讲给谁听）：{audience or '普通听众，不特别限定'}\n"
            f"你的视角：{name}\n"
            f"{style_line}"
            f"字数范围：{word_limit}~{word_limit + 40} 字\n\n"
            f"请从「{name}」这个视角，为上面的讲解对象写一段口语讲解。"
        )
        result = await Runner.run(agent, prompt)
        return result.final_output.strip()

    async def _orchestrate(
        self,
        topic: str,
        perspectives: list[str],
        sections: list[str],
        total_words: int,
        style_desc: str = "",
        audience: str = "",
    ) -> str:
        joined = "\n\n".join(
            f"【{name}】\n{content}" for name, content in zip(perspectives, sections)
        )
        style_line = f"整体文案风格：{style_desc}（开场、过渡、收尾都要贴合这个风格）\n" if style_desc else ""
        prompt = (
            f"讲解对象：{topic}\n"
            f"面向听众：{audience or '普通听众，不特别限定'}（开场称呼与用词深浅要贴合听众）\n"
            f"目标总字数：约 {total_words} 字\n"
            f"{style_line}"
            f"顺序：{' → '.join(perspectives)}\n\n"
            f"以下是各视角专家写好的分段内容，请编排成一篇连贯的完整语音讲解稿：\n\n{joined}"
        )
        result = await Runner.run(self.agents["orchestrator"], prompt)
        return result.final_output.strip()
