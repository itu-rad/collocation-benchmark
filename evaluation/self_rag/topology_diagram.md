flowchart LR
load_sched["`PoissonLoadScheduler
{
&emsp;'max_queries': 3,
&emsp;'timeout': 600000,
&emsp;'config': {
&emsp;&emsp;'rate': 0.1
&emsp;}
}`"]
style load_sched text-align:left
subgraph Stages
0["`Question dataloader
{
&emsp;'id': 0,
&emsp;'name': 'Question dataloader',
&emsp;'component': 'stages.self_rag.SelfRAGDataLoader',
&emsp;'outputs': [
&emsp;&emsp;1
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'batch_size': 1,
&emsp;&emsp;'dataset': {
&emsp;&emsp;&emsp;'name': 'RUC-NLPIR/FlashRAG_datasets',
&emsp;&emsp;&emsp;'subset': 'web_questions',
&emsp;&emsp;&emsp;'split': 'test',
&emsp;&emsp;&emsp;'question_column': 'question'
&emsp;&emsp;}
&emsp;}
}`"]
style 0 text-align:left
0 --> 1
1["`Document retriever
{
&emsp;'id': 1,
&emsp;'name': 'Document retriever',
&emsp;'component': 'stages.self_rag.ChromaRetriever',
&emsp;'outputs': [
&emsp;&emsp;2
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.FirstSubmittedPolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'corpus_dataset': {
&emsp;&emsp;&emsp;'name': 'yahma/alpaca-cleaned',
&emsp;&emsp;&emsp;'split': 'train',
&emsp;&emsp;&emsp;'text_column': 'output',
&emsp;&emsp;&emsp;'max_docs': 5000
&emsp;&emsp;},
&emsp;&emsp;'top_k': 3,
&emsp;&emsp;'collection_name': 'self_rag_decomposed_corpus'
&emsp;}
}`"]
style 1 text-align:left
1 --> 2
2["`Retrieval grader formatter
{
&emsp;'id': 2,
&emsp;'name': 'Retrieval grader formatter',
&emsp;'component': 'stages.self_rag.RetrievalGraderFormatter',
&emsp;'outputs': [
&emsp;&emsp;3
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'tokenizer_stage_id': 3
&emsp;}
}`"]
style 2 text-align:left
2 --> 3
3["`Grader LLM
{
&emsp;'id': 3,
&emsp;'name': 'Grader LLM',
&emsp;'component': 'stages.llm_mlx.Inference',
&emsp;'outputs': [
&emsp;&emsp;4
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'model': {
&emsp;&emsp;&emsp;'name': 'mlx-community/Llama-3.2-1B-Instruct-4bit',
&emsp;&emsp;&emsp;'gen_kwargs': {
&emsp;&emsp;&emsp;&emsp;'max_tokens': 16
&emsp;&emsp;&emsp;}
&emsp;&emsp;}
&emsp;}
}`"]
style 3 text-align:left
3 --> 4
4["`Grade router
{
&emsp;'id': 4,
&emsp;'name': 'Grade router',
&emsp;'component': 'stages.self_rag.BinaryRouter',
&emsp;'outputs': [
&emsp;&emsp;5,
&emsp;&emsp;10,
&emsp;&emsp;12
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'max_retries': 2,
&emsp;&emsp;'retry_is_yes': false,
&emsp;&emsp;'yes_stage_id': 5,
&emsp;&emsp;'no_stage_id': 10,
&emsp;&emsp;'end_stage_id': 12
&emsp;}
}`"]
style 4 text-align:left
4 --> 5
4 --> 10
4 --> 12
5["`Answer generator formatter
{
&emsp;'id': 5,
&emsp;'name': 'Answer generator formatter',
&emsp;'component': 'stages.self_rag.AnswerGeneratorFormatter',
&emsp;'outputs': [
&emsp;&emsp;6
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.FirstSubmittedPolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'tokenizer_stage_id': 6
&emsp;}
}`"]
style 5 text-align:left
5 --> 6
6["`Generator LLM
{
&emsp;'id': 6,
&emsp;'name': 'Generator LLM',
&emsp;'component': 'stages.llm_mlx.Inference',
&emsp;'outputs': [
&emsp;&emsp;7
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'model': {
&emsp;&emsp;&emsp;'name': 'mlx-community/Llama-3.2-1B-Instruct-4bit',
&emsp;&emsp;&emsp;'gen_kwargs': {
&emsp;&emsp;&emsp;&emsp;'max_tokens': 256
&emsp;&emsp;&emsp;}
&emsp;&emsp;}
&emsp;}
}`"]
style 6 text-align:left
6 --> 7
7["`Hallucination grader formatter
{
&emsp;'id': 7,
&emsp;'name': 'Hallucination grader formatter',
&emsp;'component': 'stages.self_rag.HallucinationGraderFormatter',
&emsp;'outputs': [
&emsp;&emsp;8
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'tokenizer_stage_id': 8
&emsp;}
}`"]
style 7 text-align:left
7 --> 8
8["`Hallucination LLM
{
&emsp;'id': 8,
&emsp;'name': 'Hallucination LLM',
&emsp;'component': 'stages.llm_mlx.Inference',
&emsp;'outputs': [
&emsp;&emsp;9
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'model': {
&emsp;&emsp;&emsp;'name': 'mlx-community/Llama-3.2-1B-Instruct-4bit',
&emsp;&emsp;&emsp;'gen_kwargs': {
&emsp;&emsp;&emsp;&emsp;'max_tokens': 16
&emsp;&emsp;&emsp;}
&emsp;&emsp;}
&emsp;}
}`"]
style 8 text-align:left
8 --> 9
9["`Hallucination router
{
&emsp;'id': 9,
&emsp;'name': 'Hallucination router',
&emsp;'component': 'stages.self_rag.BinaryRouter',
&emsp;'outputs': [
&emsp;&emsp;5,
&emsp;&emsp;12,
&emsp;&emsp;12
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'max_retries': 2,
&emsp;&emsp;'retry_is_yes': true,
&emsp;&emsp;'yes_stage_id': 5,
&emsp;&emsp;'no_stage_id': 12,
&emsp;&emsp;'end_stage_id': 12
&emsp;}
}`"]
style 9 text-align:left
9 --> 5
9 --> 12
9 --> 12
10["`Query rewrite formatter
{
&emsp;'id': 10,
&emsp;'name': 'Query rewrite formatter',
&emsp;'component': 'stages.self_rag.QuestionRewriterFormatter',
&emsp;'outputs': [
&emsp;&emsp;11
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'tokenizer_stage_id': 3
&emsp;}
}`"]
style 10 text-align:left
10 --> 11
11["`Query rewrite LLM
{
&emsp;'id': 11,
&emsp;'name': 'Query rewrite LLM',
&emsp;'component': 'stages.llm_mlx.Inference',
&emsp;'outputs': [
&emsp;&emsp;1
&emsp;],
&emsp;'polling_policy': 'utils.queues.polling.SingleQueuePolicy',
&emsp;'disable_logs': false,
&emsp;'config': {
&emsp;&emsp;'model': {
&emsp;&emsp;&emsp;'name': 'mlx-community/Llama-3.2-1B-Instruct-4bit',
&emsp;&emsp;&emsp;'gen_kwargs': {
&emsp;&emsp;&emsp;&emsp;'max_tokens': 128
&emsp;&emsp;&emsp;}
&emsp;&emsp;},
&emsp;&emsp;'depends_on_id': 3
&emsp;}
}`"]
style 11 text-align:left
11 --> 1
12["`End stage
{
&emsp;'id': 12,
&emsp;'name': 'End stage',
&emsp;'component': 'stages.Stage',
&emsp;'outputs': [],
&emsp;'polling_policy': 'utils.queues.polling.FirstSubmittedPolicy',
&emsp;'disable_logs': false,
&emsp;'config': {}
}`"]
style 12 text-align:left
end
load_sched -->|queue depth:5|0
12 --> load_sched
