---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# AI Agent for Hugging Face GAIA benchmark

This code is for the final project of Hugging Face AI Agent course. The purpose is to build an agent and evaluate its performance using a subset of the GAIA benchmark.

The implemented solution got 75% score on evaulation subset of this benchmark. It is possible to improve further by fixing the prompts to ensure the simplified final answer (e.g. it should be "89706.00" not "$89706.00"). 

# Models and Tools

- `claude-sonnet-4-6` (via Anthropic Claude API)
- Tavily Search (via Tavily API) for websearch
- duckduckgo search for websearch as backup for Tavily
- openai-whisper for audio files
- youtube-transcript-api for youtube videos
- PyPDF2 for PDF file
- pandas for Excel file
- base64 to encode image and attached to Claude as vision (hence, no need for OCR tools)
