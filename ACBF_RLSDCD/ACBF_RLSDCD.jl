include("../matrix_operations/ACBF_operations.jl")


export fACBF_DCD!
export fACBF_RLSDCD!


function fACBF_DCD!(M ::Int64, 
            H ::Float64, 
            B ::Int64, 
            Nu ::Int64,
            Rϕ ::ACBF_Matrix,  
            res ::Array{Float64,1},
            Δw ::Array{Float64,1}) 

    # The arguments Δw and β are also the outputs.  β will contain the residue on exit

    h = H/2
    b = 1
    Δw .= zeros(M)

    @inbounds for k=1:Nu
        p = argmax(abs.(res))
        pp = @inbounds mod1(Rϕ.cb_head[1] + p-1, Rϕ.diag_num_elements[1])

        while @inbounds abs(res[p]) ≤ (h/2) * Rϕ.dmtrx[pp] # Equivalent to [Rϕ.top[1]+pp], since Rϕ.top[1]=0 always.
            b=b+1
            h=h/2

            if b > B
                break
            end #if

        end #while

        @inbounds δwp=(sign(res[p])*h)
        @inbounds Δw[p]=Δw[p]+δwp
        
        
        update_ACBF_Vector!(res, Rϕ, δwp, p)

    end #for k=1:Nu

    return
end

function fACBF_RLSDCD!(w ::Array{Float64,1},
                    e ::Array{Float64,1},
                    hi ::Array{Float64,1}, 
                    x ::Array{Float64,1}, 
                    s::Array{Float64,1},
                    M::Int64, 
                    λrls::Float64, 
                    δ::Float64, 
                    Nu::Int64)

    Nit=length(s)
    w.=zeros(M)
    β=zeros(M)
    Δw=zeros(M)
    y=1.0

    R = initialize_ACBF_Matrix(M, δ)


    @inbounds for n=1:Nit
        u=@view x[n+M-1:-1:n]
        y=u'*w           # dot product
        @inbounds e[n]=s[n]-y

        update_ACBF_Matrix!(R, u, λrls)
        
        # the macro @. is provided to convert every function call, operation, and assignment in an expression into the "dotted" version.
        @. β = @inbounds λrls * β + e[n] * u

        fACBF_DCD!(M, 4.0, 16, Nu, R, β, Δw) 
        @inbounds @. w = w + Δw
    end
    return true
end

