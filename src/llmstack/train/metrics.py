def perplexity(loss: float) -> float:
    import math
    return math.exp(min(20, loss))
