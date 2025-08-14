# 🎬 Video Transcriber

Command-line tool that takes a video or audio file as input and produces the audio transcript as raw text (.txt) and subtitles (.srt). For foreign-language speech, you can either translate to English or transcribe in the original language.

## Features

- Can generate `.srt` subtitles with synchronized timestamps.
- Can translate foreign-language speech and generate transcripts in English.
- No restriction on recording length. Even though the OpenAI API limits uploads to **25 MB**, the tool automatically splits larger recordings into smaller chunks, processes them individually, and merges the results.
- Works with both audio and video files. For videos, the audio is extracted automatically using `ffmpeg`.
- Supports all common formats (e.g., `*.{mp3,wav,m4a,mp4,mov,aac,flac,ogg,mkv}`).
- Translation and subtitle generation are only supported when using the `whisper-1` model. For same-language transcription without subtitles, `gpt-4o-transcribe` or `gpt-4o-mini-transcribe` are also good options.

- Includes a batch workflow (`batch_transcribe.py`) to process multiple recordings in parallel for efficient large-scale transcription.

## Setup

You need python, an openai api key and `ffmpeg` (handles audio extraction).

Create a Python virtual environment and install the required packages

```bash
pip install -r requirements.txt
```

Set your OpenAI API key in your shell before running

```bash
export OPENAI_API_KEY=your_api_key_here
```

## How to run

To transcribe a video or audio file to a text file simply run

```bash
python transcribe.py file.mp4
```

You can change the transcription model, generate subtitles and translate, with the following options

- `--output`: Set the output directory. Defaults to the current directory.
- `--model`: The transcription model to use. Defaults to `whisper-1`.
- `--srt`: Generate an `.srt` subtitle file.
- `--translate`: Generate the transcript in English.

Use `batch_transcribe.py` to process many files at once with parallel execution

```bash
python batch_transcribe.py videos_folder/ --output-dir transcripts_folder/
```
