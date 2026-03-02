from itertools import islice, tee


def compute_ngrams(list_of_words: list[str], n_gram_size: int) -> list[tuple[str]]:
    """Compute n-grams from a list of words."""
    list_to_check = list_of_words
    n_grams = []
    while True:
        a, b = tee(list_to_check)
        n_gram = tuple(islice(a, n_gram_size))
        if len(n_gram) == n_gram_size:
            n_grams.append(n_gram)
            next(b)
            list_to_check = b
        else:
            break
    return n_grams
