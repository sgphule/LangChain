from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def load_transcript(video_id: str) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        return " ".join(snippet.text for snippet in fetched_transcript)
    except TranscriptsDisabled:
        return ""