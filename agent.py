"""四个视角专家 agent + 一个总控 agent 的定义。

改造自 awesome-llm-apps 的 ai_audio_tour_agent：
- 旅游四维度(历史/建筑/文化/美食) → 通用讲解四视角(背景/看点/深度/延伸)，多行业复用
- 模型 gpt-4o → DeepSeek（OpenAI 兼容，走 ChatCompletions）
- 去掉 OpenAI 联网工具（讲解稿用模型自身知识即可，省成本/延时）
- 去掉 pydantic 结构化输出，改纯文本（DeepSeek 经 SDK 更稳）
"""

from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled

# 关掉 Agents SDK 默认的 OpenAI trace 上报（否则用 DeepSeek 会报错）
set_tracing_disabled(True)


def make_model(api_key: str) -> OpenAIChatCompletionsModel:
    """用 DeepSeek 的 OpenAI 兼容端点构建模型。"""
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=api_key)
    return OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=client)


# 四个视角的公共约束（都会拼到各自 instructions 后面）
_COMMON = """
输出要求（严格遵守）：
- 直接给讲解正文，不要任何标题、小标题、序号、markdown 符号
- 纯口语，像一个讲解员在现场娓娓道来，会被直接送进语音合成，所以要自然、顺口、不能有书面符号
- 不要编造事实；不确定的内容一笔带过，不要展开成假细节
- 字数严格控制在给定范围内
- 按「面向听众」调整深浅：给孩子讲要浅显、多打比方；给专业听众可直接用术语
- 你的任务是「讲解」这个对象本身，不是替它创作内容（例如对象是"童话故事"时，
  你要讲解童话故事这个事物，而不是编一个童话）
"""

PERSPECTIVES = {
    "背景介绍": (
        "你是「背景介绍」视角的讲解专家。给定一个讲解对象，你负责讲清楚：它是什么、"
        "属于什么类别、大致的来历和概况，让听众先建立整体印象。" + _COMMON
    ),
    "核心看点": (
        "你是「核心看点」视角的讲解专家。给定一个讲解对象，你负责挑出最值得关注的亮点、"
        "特色和与众不同之处，告诉听众重点看什么、为什么值得关注。" + _COMMON
    ),
    "深度解读": (
        "你是「深度解读」视角的讲解专家。给定一个讲解对象，你负责讲背后的故事、原理、"
        "意义或冷知识，让听众听到表面之外的东西，产生「原来如此」的感觉。" + _COMMON
    ),
    "实用延伸": (
        "你是「实用延伸」视角的讲解专家。给定一个讲解对象，你负责给出实用信息、使用/欣赏建议、"
        "相关延伸或注意事项，让听众听完能用得上、能继续了解。" + _COMMON
    ),
}

ORCHESTRATOR_INSTRUCTIONS = """
你是「总控」讲解 agent，把几位视角专家写好的分段内容，编排成一篇完整、连贯的语音讲解稿。
你的任务：
1. 开头写一句自然温暖的欢迎/引入
2. 按给定顺序把各段内容串起来，段落之间用自然的口语过渡（比如「说完这些，我们再看看…」），
   不要改动各段的核心内容，只做衔接
3. 结尾写一句简短收束
输出要求：纯口语正文，不要任何标题、序号、markdown 符号，通篇像一个讲解员一口气讲下来，
会被直接送进语音合成。总字数控制在给定目标附近。
"""


def build_agents(model: OpenAIChatCompletionsModel) -> dict:
    """运行时用给定模型构建全部 agent（key 运行时才有，不能在 import 时建）。"""
    perspective_agents = {
        name: Agent(name=f"{name}Agent", instructions=instr, model=model)
        for name, instr in PERSPECTIVES.items()
    }
    orchestrator = Agent(
        name="OrchestratorAgent", instructions=ORCHESTRATOR_INSTRUCTIONS, model=model
    )
    return {"perspectives": perspective_agents, "orchestrator": orchestrator}
