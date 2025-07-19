import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

from src.content.sentiment.abstract import AbstractContentSentimentType
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import ContentType, SentimentType
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


class NLTKVaderContentSentimentType(AbstractContentSentimentType):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        nltk.download("vader_lexicon")

    def get_sentiment_type(self) -> SentimentType:
        with Session(self.engine) as session:
            return session.execute(
                select(SentimentType).where(SentimentType.name == "NLTKVader")
            ).scalar_one()

    def get_content_types(self) -> list[ContentType]:
        with Session(self.engine) as session:
            return list(
                session.execute(
                    select(ContentType).where(ContentType.name == "NEWS")
                ).scalars()
            )

    @property
    def columns(self) -> list[str]:
        return [
            "sentiment_score",
            "positive_sentiment_score",
            "negative_sentiment_score",
            "neutral_sentiment_score",
        ]

    def _get_sentiment_data(self, content: pd.Series) -> pd.DataFrame:
        # Initialize the sentiment analyzer.
        sentiment_analyzer = SentimentIntensityAnalyzer()

        # Get the sentiment data for a single content.
        frame = pd.DataFrame(
            content.apply(
                lambda x: sentiment_analyzer.polarity_scores(str(x))
            ).to_list()
        )

        # Rename the columns.
        frame.rename(
            columns={
                "neg": "negative_sentiment_score",
                "neu": "neutral_sentiment_score",
                "pos": "positive_sentiment_score",
                "compound": "sentiment_score",
            },
            inplace=True,
        )

        return frame
