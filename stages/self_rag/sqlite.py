import sqlite3
import os

from stages.stage import Stage, log_phase
from utils.schemas import StageModel, PipelineModel, Query


class SQLiteSearch(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        # create a connection to the database
        con = sqlite3.connect(self.extra_config["db_path"])
        self._cur = con.cursor()

    def get_schema(self):
        # return the schema of the database
        tables = self._cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return tables

    @log_phase
    def prepare(self):
        super().prepare()

        for table_name, sql in self.get_schema():
            print(f"Table: {table_name}")
            print("Schema", sql)

    def pre_run(self) -> None:
        # connection to the database needs to be reeastablished in the run thread
        con = sqlite3.connect(self.extra_config["db_path"])
        self._cur = con.cursor()

    def run(self, query: Query) -> dict[int, Query]:
        # execute the query and return the results

        print("query:\n", query.data)
        results = self._cur.execute(query.data[0]).fetchall()

        print("results:\n", results)

        query.data = results

        output = {idx: query for idx in self.output_queues}
        return output
