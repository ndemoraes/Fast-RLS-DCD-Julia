include("../matrix_operations/JCBF_operations.jl")
export fJCBF_DCD!
export fJCBF_RLSDCD!

## Uses Julia circular buffer
function fJCBF_DCD!(M ::Int64, H ::Float64, B ::Int64, Nu ::Int64,
    Rϕ ::Array{CircularBuffer{Float64},1}, β ::Array{Float64,1},
    Δw ::Array{Float64,1})
    # The arguments Δw and β are also the outputs.  β will contain the residue on exit
    h = H/2
    b = 1
    res = β # Note that this only copies the address - changing res will change β
    Δw .= zeros(M) # Zeroes Δw without allocating new vector
    @inbounds for k=1:Nu
        p = argmax(abs.(res))
        while @inbounds abs(res[p]) ≤ (h/2)*Rϕ[1][p]
            b=b+1
            h=h/2
            if b > B
                break
            end
        end
        @inbounds fact=sign(res[p])*h
        @inbounds Δw[p]=Δw[p]+fact

        update_JCBF_Matrix!(Rϕ, res, fact)

    end # for
    return
end

## Uses Julia Circular Buffer
function fJCBF_RLSDCD!(w ::Array{Float64,1},
    e ::Array{Float64,1},
    hi ::Array{Float64,1}, 
    x ::Array{Float64,1}, 
    s::Array{Float64,1},
    M::Int64, 
    λrls::Float64, 
    δ::Float64, 
    Nu::Int64)

    β=zeros(M)
    Δw=zeros(M)
    Nit=length(s)
    w.=zeros(M)

    R = initialize_JCBF_Matrix(M, δ)

    @inbounds for n=1:Nit
        @inbounds u=@view x[n+M-1:-1:n]
        y=u'*w            # dot product
        @inbounds e[n]=s[n]-y

        update_JCBF_Matrix!(R, u, λrls)

        @inbounds @. β = λrls * β + e[n] * u
        fJCBF_DCD!(M, 4.0, 16, Nu, R, β, Δw)
        @. w = w + Δw

    end
    return true
end


function fJCBF_RLSDCD(hi ::Array{Float64,1}, 
                x ::Array{Float64,1}, 
                s::Array{Float64,1},
                M::Int64, 
                λrls::Float64, 
                δ::Float64, 
                Nu::Int64)
                w=zeros(M)
                β=zeros(M)
                Δw=zeros(M)


    R = initialize_JCBF_Matrix(M, δ)

    Nit=length(s)
    e=zeros(Nit);
    MSD=zeros(Nit);
    @inbounds for n=1:Nit
        @inbounds u=@views x[n+M-1:-1:n]
        y=u⋅w            # dot product
        @inbounds e[n]=s[n]-y

        update_JCBF_Matrix!(R, u, λrls)

        @inbounds @. β = λrls * β + e[n] * u
        fJCBF_DCD(M, 4.0, 16, Nu, R, β, Δw)
        @. w = w + Δw

        @inbounds MSD[n]=norm(w-hi)^2
    end
    return w, e, MSD
end
