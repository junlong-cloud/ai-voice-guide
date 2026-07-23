"""AI 讲解员 · 通用语音导览生成器（Streamlit 入口）

多智能体协作(4 视角专家 + 1 总控) + 情感语音 TTS(硅基流动 CosyVoice2)。
输入任意行业的主题 → 生成口语讲解稿 → 合成带情感的语音。
"""

import asyncio
from pathlib import Path

import streamlit as st
from openai import OpenAI

from manager import VoiceGuideManager, WORDS_PER_MINUTE

# 【内容层】文案风格 → 传给写稿 agent，影响用词、句式、节奏
STYLES = {
    "标准讲解": "",
    "严谨科普": "用词准确克制，多用事实、数据和专业名词，句式完整，少用感叹和口头语，像科普纪录片解说",
    "轻松活泼": "用词生动口语，多用短句、设问和适度感叹，节奏轻快，像朋友边逛边聊",
    "故事叙述": "以故事和场景切入，多用细节与画面感描写，有起承转合，像讲一段往事",
}

# 【呈现层】朗读语气 → CosyVoice 情感前缀（<|endofprompt|> 之前是语气描述）
EMOTIONS = {
    "平和讲述": "用平和、娓娓道来的语气讲述",
    "热情洋溢": "用热情、亲切、有感染力的语气讲述",
    "兴奋活泼": "用兴奋、活泼、充满好奇的语气讲述",
    "沉稳权威": "用沉稳、专业、有权威感的语气讲述",
}

VOICES = {
    "anna（女声·温柔）": "anna",
    "bella（女声·亲和）": "bella",
    "alex（男声·沉稳）": "alex",
    "benjamin（男声·磁性）": "benjamin",
}


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


def synthesize(text: str, sf_key: str, voice: str, emotion_desc: str) -> Path:
    """硅基流动 CosyVoice2 情感语音合成。"""
    client = OpenAI(api_key=sf_key, base_url="https://api.siliconflow.cn/v1")
    out_path = Path(__file__).parent / "speech_guide.mp3"
    # 情感控制：<|endofprompt|> 前是语气描述，后是正文
    payload = f"{emotion_desc}<|endofprompt|>{text}"
    with client.audio.speech.with_streaming_response.create(
        model="FunAudioLLM/CosyVoice2-0.5B",
        voice=f"FunAudioLLM/CosyVoice2-0.5B:{voice}",
        input=payload,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out_path)
    return out_path


st.set_page_config(page_title="AI 讲解员 · 语音导览生成器", page_icon="🎧", layout="wide")

# 隐藏 Streamlit 自带的工具栏 / 菜单 / 页脚品牌信息
st.markdown(
    """
    <style>
      /* 只隐藏 Deploy 按钮 / 三点菜单 / 页脚品牌。
         注意：不能整块隐藏 stToolbar —— 侧边栏展开按钮(stExpandSidebarButton)就在里面 */
      [data-testid="stAppDeployButton"],
      [data-testid="stMainMenuButton"],
      [data-testid="stMainMenu"],
      #MainMenu,
      footer,
      [data-testid="stDecoration"] {
          display: none !important;
      }
      /* 保险：确保侧边栏展开按钮始终可用 */
      [data-testid="stExpandSidebarButton"] { display: inline-flex !important; }
      .block-container { padding-top: 2.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🔑 API 配置")
    deepseek_key = st.text_input("DeepSeek API Key", type="password")
    siliconflow_key = st.text_input("硅基流动 API Key", type="password")
    st.caption("DeepSeek 负责多 agent 写稿，硅基流动 CosyVoice2 负责情感语音合成。")

st.title("🎧 AI 讲解员 · 通用语音导览生成器")
st.info(
    "输入**任意主题**（景点 / 展品 / 一款产品 / 一个概念都行），"
    "4 个视角专家 Agent 协作写讲解稿，再用**情感语音**念出来。\n\n"
    "多智能体协作 · 情感语音 TTS · 多行业复用"
)

col1, col2 = st.columns([2, 1])
with col1:
    topic = st.text_input(
        "📌 讲解对象（讲什么）· 必填",
        placeholder="例如：故宫太和殿 / 一款扫地机器人 / 什么是 RAG / 童话故事",
        help="要被讲解的那个事物本身。工具做的是「讲解」，不是替它创作内容。",
    )
    audience = st.text_input(
        "👥 面向听众（讲给谁）· 选填",
        placeholder="例如：小学生 / 行业新人 / 专业投资者 / 亲子家庭；不填则面向普通听众",
        help="影响用词深浅和举例方式。同一个对象，讲给孩子和讲给专家完全不同。",
    )
    perspectives = st.multiselect(
        "🎯 讲解视角（多 agent，每个视角一个专家）",
        options=["背景介绍", "核心看点", "深度解读", "实用延伸"],
        default=["背景介绍", "核心看点", "深度解读"],
    )
with col2:
    duration = st.slider("⏱️ 讲解时长（分钟）", 1.0, 5.0, 2.0, 0.5)
    style_label = st.selectbox(
        "✍️ 文案风格（内容层）", list(STYLES.keys()),
        help="影响 agent 怎么写：用词、句式、节奏。改这个需要重新生成讲解稿。",
    )
    emotion_label = st.selectbox(
        "😊 朗读语气（呈现层）", list(EMOTIONS.keys()),
        help="影响 TTS 怎么念。改这个只需重新合成，不用重写稿子。",
    )
    voice_label = st.selectbox("🎙️ 音色", list(VOICES.keys()))

st.caption(f"预计约 {int(duration * WORDS_PER_MINUTE)} 字，由 {len(perspectives) or 1} 个视角专家分工完成。")

# ① 多 agent 写稿
if st.button("① 生成讲解稿", type="primary"):
    if not deepseek_key:
        st.error("请先在左侧填入 DeepSeek API Key。")
    elif not topic:
        st.error("请输入讲解主题。")
    elif not perspectives:
        st.error("请至少选择一个讲解视角。")
    else:
        spinner_msg = (
            f"{len(perspectives)} 个视角专家正在按「{style_label}」风格"
            f"{'、面向「' + audience + '」' if audience else ''}协作写稿…"
        )
        with st.spinner(spinner_msg):
            mgr = VoiceGuideManager(deepseek_key)
            st.session_state["script"] = run_async(
                mgr.run(topic, perspectives, duration, STYLES[style_label], audience)
            )
            st.session_state["script_style"] = style_label
            st.session_state["script_audience"] = audience
        st.rerun()

# ② 人工确认/编辑后再合成（TTS 比文本贵，不自动烧钱；也便于人工润色）
if "script" in st.session_state:
    st.markdown("### 📝 讲解稿（可自由编辑，改完再合成语音）")
    st.text_area("讲解稿", key="script", height=320, label_visibility="collapsed")
    chars = len(st.session_state["script"])
    written_style = st.session_state.get("script_style", "标准讲解")
    written_audience = st.session_state.get("script_audience", "")
    st.caption(
        f"当前 {chars} 字 · 约 {chars / WORDS_PER_MINUTE:.1f} 分钟 · "
        f"按「{written_style}」风格写的"
        + (f" · 面向「{written_audience}」" if written_audience else " · 面向普通听众")
    )
    if written_style != style_label or written_audience != audience:
        st.warning(
            "文案风格或听众改过了（当前稿子仍是旧设定写的）——"
            "想让**文案本身**跟着变，需要重新点 ①生成讲解稿；只改朗读语气的话直接点 ② 即可。"
        )

    if st.button("② 合成情感语音", type="primary"):
        if not siliconflow_key:
            st.error("请先在左侧填入硅基流动 API Key。")
        else:
            with st.spinner("CosyVoice 正在合成情感语音…"):
                try:
                    audio_path = synthesize(
                        st.session_state["script"],
                        siliconflow_key,
                        VOICES[voice_label],
                        EMOTIONS[emotion_label],
                    )
                    st.subheader("🎧 语音讲解")
                    st.audio(str(audio_path), format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            "📥 下载音频",
                            f,
                            file_name=f"{topic or '讲解'}_讲解.mp3",
                            mime="audio/mp3",
                        )
                    st.success(
                        f"完成！当前语气：{emotion_label}。"
                        "换个情感语气再点一次合成即可对比——不用重新生成稿子。"
                    )
                except Exception as e:
                    st.error(f"语音合成失败：{e}")
