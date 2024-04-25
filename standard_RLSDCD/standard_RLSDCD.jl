include("../matrix_operations/Standard_operations.jl")
export DCD!
export RLSDCD!

# @@@@@@@@@@@@
# DCD
# calculates the residue 
# called by RLSDCD
#
# @@@@@@@@@@@@

function DCD!(M ::Int64, 
             H ::Float64, 
             B ::Int64, 
             Nu ::Int64,
             Rϕ ::Array{Float64,2}, 
             res ::Array{Float64,1}, 
             Δw ::Array{Float64,1})
    # Modifies Δw and res

    h = H/2
    b = 1
    Δw .= zeros(M)  # vector of zeros
    for k=1:Nu
        p = argmax(abs.(res)) # can be anything from 1 to M
        
        while abs(res[p]) ≤ (h/2)*Rϕ[p,p]
            b=b+1
            h=h/2
            if b > B
                break
            end
        end #while

        δwp = sign(res[p])*h
        Δw[p]=Δw[p] + δwp   # vector element, p is index

        update_vector!(res, Rϕ, δwp, p)

    end #for
    return true
end # fn DCD

function RLSDCD!(w ::Array{Float64,1},
    e ::Array{Float64,1},
    hi ::Array{Float64,1}, 
    x ::Array{Float64,1}, 
    s ::Array{Float64,1},
    M ::Int64, 
    λrls ::Float64, 
    δ ::Float64, 
    Nu ::Int64)

w.=zeros(M)
β=zeros(M)
Δw=zeros(M)

# defining and initializing the matrix
R = initialize_matrix(M, δ)

Nit=length(s)
e.=zeros(Nit)


for n=1:Nit
u=@view x[n+M-1:-1:n]
y=u'*w
e[n]=s[n]-y

update_matrix!(R, u, λrls)

@. β = λrls * β + e[n] * u   #equilavent to β[N] = λrls * res[N] + e[n] * u[N]
DCD!(M, 4.0, 16, Nu, R, β, Δw)
@. w = w + Δw

end
return true

end




