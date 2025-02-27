import duckdb
import os

from stages.stage import Stage, log_phase
from utils.schemas import StageModel, PipelineModel, Query


class DuckDBSearch(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        # create a connection to the database
        self._db = duckdb.connect()

    def get_schema(self):
        # return the schema of the database
        tables = self._db.execute("SHOW ALL TABLES").fetchdf()
        return tables

    @log_phase
    def prepare(self):
        super().prepare()

        db_path = self.extra_config["db_path"]
        db_name = db_path.split("/")[-1].split(".")[0]

        tmp_conn = duckdb.connect()
        tmp_conn.sql(f"ATTACH '{db_path}' (TYPE SQLITE); USE {db_name};")
        tables = tmp_conn.execute("SHOW TABLES").fetchall()

        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            self._db.sql(
                f"CREATE TABLE {table_name} AS SELECT * FROM sqlite_scan('{db_path}', '{table_name}');"
            )
            schema = self._db.execute(f".schema {table_name}").fetchdf()
            print("Schema", schema)
        print(f"Schema:\n{self.get_schema()}")

    def run(self, query: Query) -> dict[int, Query]:
        # execute the query and return the results
        results = self._db.execute(
            'select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Rick Hamilton" )'
        ).fetchdf()

        print("results:\n", results)

        output = {idx: query for idx in self.output_queues}
        return output
