# Collocation Benchmark

This is a benchmark to measure the end-to-end performance of (collocated) machine learning pipelines. In comparison to the other existing options, this benchmark focuses on:

- Measurement of end-to-end perfomance - the alternatives often focus on quantifying the performance of DNN inference and discounting the other parts of the pipelines, such as data loading or preprocessing, which account for a significant portion of the pipeline execution.
- Modularity - while the benchmark provides different use case scenarios, the implementation of the different stages of the pipeline is not locked. The different stages of the pipeline, such as data loading are standalone building blocks, making for easier performance evaluation.
- Colocation - many deployments can benefit from smart colocation of machine learning workloads. Our benchmark will help researchers evaluate their new ideas for colocation schedulers and resource managers by providing multiple use case scenarios for colocation as well as detailed time breakdown for each of the colocated workloads for further analysis.

## Benchmark execution

```mermaid
sequenceDiagram
    participant Benchmark
    create participant LoadGen
    Benchmark->>LoadGen: Build
    create participant Logging
    Benchmark->>Logging: Build
    loop For each Pipeline
        create participant Pipeline
        Benchmark->>Pipeline: Build
        loop For each Stage
            create participant Stage
            Pipeline->>Stage: Build
        end
        activate Benchmark
        Benchmark->>+LoadGen: Start
        LoadGen-->>Logging: Log start of the test
        loop For each query
            LoadGen->>+Pipeline: Invoke Pipeline execution
            Pipeline-->>Logging: Log start of Pipeline execution
            loop For each Stage
                Pipeline-->>Logging: Log start of Stage execution
                Pipeline->>+Stage: Invoke Stage
                Stage->>-Pipeline: Return
                Pipeline-->>Logging: Log end of Stage execution
            end
            Pipeline-->>Logging: Log end of Pipeline execution
            Pipeline->>-LoadGen: Acknowledge finishing the query
            LoadGen-->>Logging: Log end of the test
        end
        LoadGen->>-Benchmark: Return
    end
```

### Benchmark's building blocks and their responsibility

#### Benchmark

Benchmark is the entry point to our benchmark. It parses configuration files, which define the pipelines as well as general settings for other building blocks of the benchmark such as the LoadGen. Based on the configuration files, initiate three other building blocks: LoadGen, Pipeline/s and Logging.

#### LoadGen

This is the building block that issues queries to the SUT based on different policies. To make this block reusable, the LoadGen is not aware of the specific type of pipelines or datasets, but rather uses sampleIDs as an abstraction to pass to data loaders, which are the first stages of the pipelines.

##### Logging

It logs the end-to-end performance statistics of the pipeline execution, such as the run time splits of the dfferent stages of the pipeline. The logging is performed in a separate thread asynchronously in order to not interfere with the pipeline's execution.

#### Pipeline

The pipeline holds the different stages and orchestrates the communication between them, as well as, the logging of execution times of the separate stages.

#### Stage

This is the building block of the pipelines. A stage can perform tasks such as data loading, data preprocessing or model execution. The stages are separated in order to make the development of specific part of a pipeline and subsequent evaluation as easy as possible.
