from chain import chain

nl_queries_file = open('datasets_nl_queries.txt', 'r')
queries = nl_queries_file.readlines()
for query in queries:
    print(query)
    try:
        result = chain.invoke(query)
        print(result)
    except Exception as x:
        print(x)

