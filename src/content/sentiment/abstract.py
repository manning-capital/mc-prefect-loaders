from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import ContentType, SentimentType, ProviderContentSentiment


class AbstractContentSentimentType(ABC):
    def __init__(self, engine: Engine):
        self.engine = engine

    @property
    def name(self) -> str:
        """The name of the sentiment type."""
        with Session(self.engine) as session:
            return session.execute(
                select(SentimentType.name).where(
                    SentimentType.id == self.get_sentiment_type_id()
                )
            ).scalar_one()

    @property
    def description(self) -> str | None:
        """The description of the sentiment type."""
        with Session(self.engine) as session:
            return session.execute(
                select(SentimentType.description).where(
                    SentimentType.id == self.get_sentiment_type_id()
                )
            ).scalar_one_or_none()

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """The columns of the sentiment type."""
        pass

    def get_sentiment_data(self, content: pd.Series) -> pd.DataFrame:
        """
        Get the sentiment data for the given content.
        """
        data = self._get_sentiment_data(content)

        # Check if the data is empty.
        if data.empty:
            return pd.DataFrame(
                {
                    column_name: pd.Series(
                        dtype=pd.api.types.pandas_dtype(
                            ProviderContentSentiment.__table__.columns[
                                column_name
                            ].type.python_type
                        )
                    )
                    for column_name in self.columns
                }
            )

        # Check if the data contains the required columns.
        if not all(col in data.columns for col in self.columns):
            raise ValueError(
                f"The data does not contain the required columns: {', '.join(self.columns)}"
            )

        # Add the sentiment type id to the data.
        data["sentiment_type_id"] = self.get_sentiment_type_id()

        return data

    def get_content_type_ids(self) -> list[int]:
        """
        Get the content type ids for the sentiment type.
        """
        return [content_type.id for content_type in self.get_content_types()]

    def get_sentiment_type_id(self) -> int:
        """
        Get the sentiment type id.
        """
        return self.get_sentiment_type().id

    @abstractmethod
    def get_sentiment_type(self) -> SentimentType:
        """
        Get the sentiment type id.

        Returns:
            The sentiment type id.
        """
        pass

    @abstractmethod
    def get_content_types(self) -> list[ContentType]:
        """
        Get the content types for the sentiment type.
        """
        pass

    @abstractmethod
    def _get_sentiment_data(self, content: pd.Series) -> pd.DataFrame:
        """
        Get the sentiment data for the given content.

        Args:
            content: The content to get the sentiment data for.

        Returns:
            The sentiment data for the given content.
        """
        pass
