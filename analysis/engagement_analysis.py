"""
Module for analyzing YouTube video and short engagement metrics and suggesting content strategies.

This module defines data structures and functions to load video metadata,
compute engagement metrics (views, likes, comments, shares, watch time),
classify content types (video vs short), and generate recommendations
for content creation that maximizes engagement.

The code is designed to be run as a standalone analysis script or imported
into larger pipelines orchestrated by Airflow.

Example usage:
    python analysis/engagement_analysis.py --input-path data/videos.csv --output-path reports/analysis_report.json
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class VideoRecord:
    video_id: str
    title: str
    publish_date: pd.Timestamp
    duration_seconds: int
    views: int
    likes: int
    comments: int
    shares: int
    watch_time_hours: float
    tags: List[str]

def load_video_data(csv_path: str) -> pd.DataFrame:
    """Load video or short metadata from a CSV file into a DataFrame.

    The CSV is expected to have columns:
        video_id,title,publish_date,duration_seconds,views,likes,comments,shares,watch_time_hours,tags

    Args:
        csv_path: Path to a CSV file containing video metadata.

    Returns:
        A pandas DataFrame with appropriate dtypes.
    """
    df = pd.read_csv(csv_path, parse_dates=["publish_date"])
    # Ensure tags column is parsed as list if stored as pipe-separated string
    if "tags" in df.columns:
        df["tags"] = df["tags"].fillna("").apply(lambda x: [tag.strip() for tag in str(x).split("|") if tag.strip()])
    return df

def compute_engagement_score(row: pd.Series) -> float:
    """Compute an engagement score for a single video record.

    The score is a weighted combination of likes, comments, shares, and average watch time.
    Adjust weights based on your business priorities.

    Args:
        row: A pandas Series representing a video record.

    Returns:
        A float engagement score.
    """
    # Avoid division by zero
    views = max(row["views"], 1)
    like_ratio = row["likes"] / views
    comment_ratio = row["comments"] / views
    share_ratio = row["shares"] / views
    watch_time_ratio = row["watch_time_hours"] * 3600 / max(row["duration_seconds"], 1)
    # Weighted sum; tune weights as needed
    return 0.4 * like_ratio + 0.3 * comment_ratio + 0.2 * share_ratio + 0.1 * watch_time_ratio

def analyze_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Add engagement scores to the DataFrame and return it sorted by score descending.

    Args:
        df: DataFrame of video records.

    Returns:
        DataFrame with an additional 'engagement_score' column sorted in descending order.
    """
    df = df.copy()
    df["engagement_score"] = df.apply(compute_engagement_score, axis=1)
    return df.sort_values("engagement_score", ascending=False)

def suggest_content_topics(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """Suggest content topics based on top performing videos.

    This function extracts tags from the top N videos by engagement
    and returns the most frequent tags as recommendations.

    Args:
        df: DataFrame with video records and engagement scores.
        top_n: Number of top videos to consider.

    Returns:
        A list of recommended tags/topics.
    """
    top_videos = df.nlargest(top_n, "engagement_score")
    tag_counts: Dict[str, int] = {}
    for tags in top_videos["tags"]:
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    # Sort tags by frequency descending
    sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
    return [tag for tag, count in sorted_tags]

def classify_content_type(duration_seconds: int) -> str:
    """Classify a piece of content as 'short' or 'video' based on duration.

    YouTube shorts are typically less than 60 seconds.

    Args:
        duration_seconds: Length of the video in seconds.

    Returns:
        'short' if duration is less than 60 seconds, otherwise 'video'.
    """
    return "short" if duration_seconds < 60 else "video"

def main(input_path: str) -> None:
    """Run the analysis pipeline on a local CSV of YouTube video data.

    Reads the data, computes engagement scores, prints the top performing videos,
    and suggests content topics.

    Args:
        input_path: Path to the input CSV file.
    """
    df = load_video_data(input_path)
    df["content_type"] = df["duration_seconds"].apply(classify_content_type)
    df = analyze_engagement(df)
    print("Top 10 videos by engagement:")
    print(df[["video_id", "title", "engagement_score", "content_type"]].head(10))
    suggestions = suggest_content_topics(df, top_n=10)
    print("Recommended topics to explore based on top videos:")
    for topic in suggestions:
        print(f"- {topic}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze YouTube videos and suggest content topics.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the CSV file containing video metadata.")
    args = parser.parse_args()
    main(args.input_path)
