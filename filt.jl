function filt_(h, x)
    N = length(x)
    M = length(h)
    y = zeros(N)
    for n in 1:N
        y[n] = h[1] * x[n]
        for k in 1:min(n-1, M-1)
            y[n] += h[k+1] * x[n-k]
        end
    end
    return y
end

