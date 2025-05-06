import os
import re
import requests
import pandas as pd
import isodate
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from googleapiclient.discovery import build

pd.options.display.float_format = '{:.0f}'.format

class YouTubeShortsAnalyzer:
    def __init__(self, env_file="youtubekey.env", max_results=25):
        load_dotenv(env_file)  # load environment variables
        self.api_key = os.getenv("youtubekey")
        if not self.api_key:
            raise ValueError("API Key not found. Check your youtubekey.env file.")
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        self.max_results = max_results
        self.all_shorts_df = pd.DataFrame()

    def _fetch_niche_shorts(self, niche, days_limit=None):
        """
        Fetch Shorts for a specific niche. 
        If days_limit is provided, uses the 'publishedAfter' 
        parameter to only fetch videos from that many days ago.
        """
        # If user specifies days_limit, compute the publishedAfter date
        if days_limit is not None:
            published_after_date = (datetime.now(timezone.utc) - timedelta(days=days_limit))
            published_after_iso = published_after_date.isoformat()
        else:
            published_after_iso = None

        # Build the search query
        search_args = {
            "q": niche,
            "part": "snippet",
            "type": "video",
            "maxResults": self.max_results
        }
        # If we have a date limit, include 'publishedAfter' in the request
        if published_after_iso:
            search_args["publishedAfter"] = published_after_iso

        # Perform the search
        search_response = self.youtube.search().list(**search_args).execute()

        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
        if not video_ids:
            return pd.DataFrame()

        # Fetch video details (snippet, statistics, contentDetails)
        details_response = self.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids)
        ).execute()

        videos_data = []
        for item in details_response.get("items", []):
            video_id = item["id"]
            duration_str = item["contentDetails"]["duration"]
            duration_td = isodate.parse_duration(duration_str)
            duration_seconds = duration_td.total_seconds()

            # Construct Shorts URL
            shorts_url = f"https://www.youtube.com/shorts/{video_id}"

            # Try to verify if it's a Shorts by redirect or fallback to duration
            try:
                r = requests.get(shorts_url, allow_redirects=True, timeout=5)
                if r.url.startswith("https://www.youtube.com/shorts/"):
                    video_type = "shorts"
                else:
                    video_type = "longs"
            except:
                video_type = "shorts" if duration_seconds <= 180 else "longs"

            # Skip non-shorts
            if video_type != "shorts":
                continue

            published_at_str = item["snippet"]["publishedAt"]
            published_at = isodate.parse_datetime(published_at_str)
            current_time = datetime.now(timezone.utc)
            days_since_upload = (current_time - published_at).days

            view_count = int(item["statistics"].get("viewCount", 0))

            videos_data.append({
                "Niche": niche,
                "Video ID": video_id,
                "Video Type": video_type,
                "Title": item["snippet"]["title"],
                "Description": item["snippet"]["description"],
                "Published Date": published_at_str,
                "Days Since Upload": days_since_upload,
                "View Count": view_count,
                "Video URL": shorts_url,
            })

        # Create DataFrame
        df = pd.DataFrame(videos_data)
        if not df.empty:
            # Sort by highest views
            df = df.sort_values("View Count", ascending=False).reset_index(drop=True)
            # Mark the top video (row 0) as "Yes"
            df["Top Video"] = ["Yes" if i == 0 else "No" for i in range(len(df))]
        return df

    def collect_data(self, niche_list, days_limit=None):
        """
        Collect data for each niche. If days_limit is provided,
        we'll restrict our search to that many days from the current date.
        """
        all_dfs = []
        for niche in niche_list:
            print(f"Collecting data for niche: {niche} ...")
            niche_df = self._fetch_niche_shorts(niche, days_limit)
            if not niche_df.empty:
                all_dfs.append(niche_df)
        if all_dfs:
            self.all_shorts_df = pd.concat(all_dfs, ignore_index=True)
            print("Data collection complete!")
        else:
            print("No Shorts data found for these niches.")

    def analyze_data(self):
        if self.all_shorts_df.empty:
            print("No data to analyze. Run collect_data() first.")
            return

        # Basic stats by niche
        niche_stats = (
            self.all_shorts_df
            .groupby("Niche")["View Count"]
            .agg(["count", "mean", "max"])
            .reset_index()
        )
        print("\n=== Niche Statistics ===")
        print(niche_stats)

        # Top video per niche
        top_videos = self.all_shorts_df[self.all_shorts_df["Top Video"] == "Yes"]
        print("\n=== Top Video by Niche ===")
        for niche in niche_stats["Niche"]:
            subset = top_videos[top_videos["Niche"] == niche]
            if not subset.empty:
                row = subset.iloc[0]
                print(f"Niche: {niche}")
                print(f"  Title: {row['Title']}")
                print(f"  Views: {row['View Count']}")
                print(f"  URL:   {row['Video URL']}\n")

    def print_top_videos_for_niche_last_week(self, niche):
        """
        OPTIONAL HELPER:
        Filter the all_shorts_df to show only a specific niche
        and only those uploaded in the last 7 days, sorted by views.
        """
        if self.all_shorts_df.empty:
            print("No data to filter. Make sure you run collect_data() first.")
            return

        # Filter only that niche + only last 7 days
        subset = self.all_shorts_df[
            (self.all_shorts_df["Niche"] == niche)
            & (self.all_shorts_df["Days Since Upload"] <= 7)
        ].copy()

        if subset.empty:
            print(f"No Shorts found for niche '{niche}' in the last 7 days.")
            return

        # Sort by highest views
        subset = subset.sort_values("View Count", ascending=False)
        # Print the top few results
        print(f"Top Shorts in the LAST WEEK for niche: {niche}")
        print(subset[["Title", "View Count", "Days Since Upload", "Video URL"]].head(10))

    def run(self, niche_list, days_limit=None):
        self.collect_data(niche_list, days_limit=days_limit)
        self.analyze_data()


if __name__ == "__main__":
    # Example usage:
    analyzer = YouTubeShortsAnalyzer(env_file="youtubekey.env", max_results=25)
    
    # Provide your niche list
    my_niches = [
        "AI art shorts",
        "Generative AI shorts",
        "AI video shorts"
    ]
    
    # Here, specify the number of days from today to search
    # e.g., last 7 days
    analyzer.run(my_niches, days_limit=7)
    
    # You can also separately call the optional helper method
    # if you want to see top videos in the last week for a single niche:
    analyzer.print_top_videos_for_niche_last_week("AI art shorts")
