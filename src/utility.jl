function logspace(start, stop, length)
    return exp2.(range(log2(start), log2(stop); length=length))
end
