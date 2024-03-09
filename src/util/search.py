from duckduckgo_search import DDGS

# DDG query wrapper
def results(query: str, n_max=1, include_all=False):
    results = DDGS().text(query, max_results=n_max)
    if include_all:
        return results
    return [e['body'] for e in results]

# Unwrap single result
def result_single(query: str) -> str:
    tmp = results(query, n_max=1, include_all=False)
    if len(tmp):
        return tmp[0]
    return None

