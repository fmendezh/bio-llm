from chain import agent_executor

nl_queries_file = open('datasets_nl_queries.txt', 'r')
queries = nl_queries_file.readlines()
# Strips the newline character
for query in queries:
    print(query)
    try:
        result = agent_executor.invoke(query)
        print(result)
    except Exception as x:
        print(x)

