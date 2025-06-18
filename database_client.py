from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class DatabaseClient:
    def __init__(self, url: str):
        self.engine = create_engine(url)

    def get_session(self) -> Session:
        return Session(self.engine)

    def close(self):
        self.engine.dispose()  # Close the engine when done
