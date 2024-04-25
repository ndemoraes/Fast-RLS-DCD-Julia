# to run echo canceling demonstration
#
# cd to directory where files are located 
# file 1 : echo_canceling.jl
# file 2 : respimp.jld
# directory with matrix operations library: ./matrix_operations
# directory with standard matrix algorithms: ./standard_RLSDCD
# directory with Julia circular buffer algorithms: ./JCBF_RLSDCD
# directory with DCBF format matrix algorithms: ./DCBF_RLSDCD

# > julia 
# julia> push!(LOAD_PATH, ".","./matrix_operations","./standard_RLSDCD","./DCBF_RLSDCD", "./JCBF_RLSDCD")
# julia> using echo_canceling
# julia> test_DCBF()
# julia> test_JCBF()
# julia> exit()
# Functions RLS!, fRLSDCD!, RLSDCD! accept vectors w and e as preallocated inputs, 
# that will also contain the output of the filter.  Funcions RLS, fRLSDCD and RLSDCD
# allocate memory for w and e internally, and return them as outputs, together with MSD

module echo_canceling
using PyPlot
using LinearAlgebra
using DSP
using JLD
using Statistics
using DataStructures
using DataFrames
using CSV
include("./standard_RLSDCD/standard_RLSDCD.jl")
include("./DCBF_RLSDCD/DCBF_RLSDCD.jl")
include("./JCBF_RLSDCD/JCBF_RLSDCD.jl")
include("./ACBF_RLSDCD/ACBF_RLSDCD.jl")

export test_JCBF
export test_DCBF
export test_ACBF
export test_RLS
export test_RLSDCD
export test_fRLSDCD
export test_fACBF_RLSDCD
export DCD!
export fDCD! 
export fRLSDCD!
export RLSDCD!
export RLS!





########


function RLSloop!(w ::Array{Float64,1}, e ::Array{Float64,1}, hi ::Array{Float64,1}, 
              x ::Array{Float64,1}, s ::Array{Float64,1}, M ::Int64, λrls ::Float64, δ ::Float64)
    w.=zeros(M)
    P=δ*diagm(0 => ones(M))
    g=zeros(M)
    Nit=length(s)
    λi=1.0/λrls
    @inbounds for n=1:Nit
        y=0.0
        γ=λrls
        @inbounds for i in 1:M
            g[i]=0.0
            @inbounds for j in 1:M                            # g .= P*u
                g[i] += P[j,i] * x[n+M-j]           # P is symmetric. It's faster to loop over fixed columns first
            end
            y += w[i]*x[n+M-i]  # y=u⋅w            # dot product
            γ += g[i]*x[n+M-1]  # g .= P*u
        end
        
        e[n]=s[n]-y
        
        γinv=1.0 / γ
        a = e[n] * γinv
        b = sqrt(abs(γinv))
        @inbounds for j in 1:M 
            w[j] += a * g[j]
            g[j] *= b
            @inbounds for i in 1:M
                P[i,j] = λi * (P[i,j] - g[i] * g[j])  #P .= λi .* (P .- (g .* transpose(g)))
            end
        end
    end
    return true
end
 
function RLS!(w ::Array{Float64,1}, e ::Array{Float64,1}, hi ::Array{Float64,1}, 
    x ::Array{Float64,1}, s ::Array{Float64,1}, M ::Int64, λrls ::Float64, δ ::Float64)
    w.=zeros(M)
    P=δ*diagm(0 => ones(M))
    g=zeros(M)
    Nit=length(s)
    λi=1.0/λrls
    @inbounds for n=1:Nit
        u=@view x[n+M-1:-1:n]
        y=u⋅w            # dot product
        @inbounds e[n]=s[n]-y
        g .= P*u
        γ=λrls+u⋅g
        γinv=1.0 / γ
        @inbounds w .= w .+ (e[n] .* γinv) .* g
        g .= g .* sqrt(abs(γinv))
        @. P = λi * (P - (g * g'))
    end
    return true
end

function RLS(hi ::Array{Float64,1}, x ::Array{Float64,1}, s ::Array{Float64,1}, M ::Int64, λrls ::Float64, δ ::Float64)
    w=zeros(M);
    P=δ*diagm(0 => ones(M));
    g=zeros(M)
    Nit=length(s)
    e=zeros(Nit);
    MSD=zeros(Nit)
    λi=1.0/λrls
    for n=1:Nit
        u=x[n+M-1:-1:n];
        y=u⋅w;            # dot product
        e[n]=s[n]-y;
        g .= P*u
        γ=λrls+u⋅g
        γinv=1.0 / γ
        w .= w .+ (e[n] .* γinv) .* g
        g .= g .* sqrt(abs(γinv))
        P .= (λi) .* (P .- (g .* g'))
        MSD[n]=norm(w-hi)^2
    end
    return w, e, MSD
end


function test_JCBF()
 print("running test_JCBF \n")
    ϵ=1.0
    σv=0.1
    λ = 0.99
    δ = 1.0


    L=100
    M=[10, 100, 1000] 
    N=5_000
    NM=length(M)
    TRLS=zeros(NM,L)
    TRLSDCD=zeros(NM,L)
    TRLSDCD4=zeros(NM,L)
    TfRLSDCD=zeros(NM,L)
    TfRLSDCD4=zeros(NM,L)

    for k=1:length(M)
        hi=randn(M[k])
        w=zeros(M[k])
        e=zeros(N)
        x=randn(N)
        s=filt(hi,1,x)+σv*randn(N)
        x=[zeros(M[k]-1); x]
        print("M= ", M[k], "\n")
        for i=1:L
            TfRLSDCD[k,i]=@elapsed fJCBF_RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 1)
            TfRLSDCD4[k,i]=@elapsed fJCBF_RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 4)
            TRLS[k,i]=@elapsed RLS!(w, e, hi, x, s, M[k], λ, δ)
            TRLSDCD[k,i]=@elapsed RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 1)
            TRLSDCD4[k,i]=@elapsed RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 4)
            print("i= ", i, "\nTempo RLS\t\t", TRLS[k,i],
            "\nTempo RLSDCD Nu = 1\t", TRLSDCD[k,i],
            "\nTempo RLSDCD Nu = 4\t", TRLSDCD4[k,i],
            "\nTempo fRLSDCD Nu = 1\t", TfRLSDCD[k,i],
            "\nTempo fRLSDCD Nu = 4\t", TfRLSDCD4[k,i], "\n\n")
        end
        print("\n")
    end
    MefRLSDCD=mean(TfRLSDCD,dims=2)[:]
    MeRLSDCD=mean(TRLSDCD,dims=2)[:]
    MefRLSDCD4=mean(TfRLSDCD4,dims=2)[:]
    MeRLSDCD4=mean(TRLSDCD4,dims=2)[:]
    MeRLS=mean(TRLS,dims=2)[:]
    MifRLSDCD=minimum(TfRLSDCD,dims=2)[:]
    MiRLSDCD=minimum(TRLSDCD,dims=2)[:]
    MifRLSDCD4=minimum(TfRLSDCD4,dims=2)[:]
    MiRLSDCD4=minimum(TRLSDCD4,dims=2)[:]
    MiRLS=minimum(TRLS,dims=2)[:]
    ResJulia = DataFrame(
        "M" => M,
        "Mean RLS" => MeRLS,
        "Mean RLSDCD1" => MeRLSDCD,
        "Mean RLSDCD4" => MeRLSDCD4,
        "Mean fRLSDCD1" => MefRLSDCD,
        "Mean fRLSDCD4" => MefRLSDCD4,
        "Min RLS" => MeRLS,
        "Min RLSDCD1" => MiRLSDCD,
        "Min RLSDCD4" => MiRLSDCD4,
        "Min fRLSDCD1" => MifRLSDCD,
        "Min fRLSDCD4" => MifRLSDCD4
    )
    CSV.write("ResultadosJuliaJ.csv", ResJulia)

    #@save "fRLSDCD.jld" MefRLSDCD MeRLSDCD MefRLSDCD4 MeRLSDCD4 MeRLS TfRLSDCD TRLSDCD TfRLSDCD4 TRLSDCD4 TRLS M L N
    #clf()
    #loglog(M,MeRLSDCD,label="RLS-DCD Nu = 1")
    #loglog(M,MeRLS,label="RLS")
    #loglog(M,MeRLSDCD4,label="RLS-DCD Nu = 4")
    #loglog(M,MefRLSDCD,label="fast RLS-DCD Nu = 1")
    #loglog(M,MefRLSDCD4,label="fast RLS-DCD Nu = 4")
    #legend()
    #grid()
    #xlabel("M")
    #title("Running time")
    #show()
    #savefig("fRLSDCD.svg")
end # test_JCBF

function test_DCBF()
    print("running test_DCBF!\n")
    ϵ=1.0
    σv=0.1
    λ = 0.99
    δ = 1.0

    L=100
    M=[10, 100, 1000] 
    N=5_000
    NM=length(M)
    TRLS=zeros(NM,L)
    TRLSDCD=zeros(NM,L)
    TRLSDCD4=zeros(NM,L)
    TfRLSDCD=zeros(NM,L)
    TfRLSDCD4=zeros(NM,L)

    for k=1:length(M)
        hi=randn(M[k])
        w=zeros(M[k])
        e=zeros(N)
        x=randn(N)
        s=filt(hi,1,x)+σv*randn(N)
        x=[zeros(M[k]);x]
        print("M= ", M[k], "\n")
        for i=1:L
            TfRLSDCD[k,i]=@elapsed fDCBF_RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 1)
            TfRLSDCD4[k,i]=@elapsed fDCBF_RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 4)
            TRLS[k,i]=@elapsed RLS!(w, e, hi, x, s, M[k], λ, δ)
            TRLSDCD[k,i]=@elapsed RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 1)
            TRLSDCD4[k,i]=@elapsed RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 4)
            print("i= ", i, "\nTempo RLS\t\t", TRLS[k,i],
            "\nTempo RLSDCD Nu = 1\t", TRLSDCD[k,i],
            "\nTempo RLSDCD Nu = 4\t", TRLSDCD4[k,i],
            "\nTempo fRLSDCD Nu = 1\t", TfRLSDCD[k,i],
            "\nTempo fRLSDCD Nu = 4\t", TfRLSDCD4[k,i], "\n\n")
        end
        print("\n")
    end
    MefRLSDCD=mean(TfRLSDCD,dims=2)[:]
    MeRLSDCD=mean(TRLSDCD,dims=2)[:]
    MefRLSDCD4=mean(TfRLSDCD4,dims=2)[:]
    MeRLSDCD4=mean(TRLSDCD4,dims=2)[:]
    MeRLS=mean(TRLS,dims=2)[:]
    MifRLSDCD=minimum(TfRLSDCD,dims=2)[:]
    MiRLSDCD=minimum(TRLSDCD,dims=2)[:]
    MifRLSDCD4=minimum(TfRLSDCD4,dims=2)[:]
    MiRLSDCD4=minimum(TRLSDCD4,dims=2)[:]
    MiRLS=minimum(TRLS,dims=2)[:]
    ResJulia = DataFrame(
        "M" => M,
        "Mean RLS" => MeRLS,
        "Mean RLSDCD1" => MeRLSDCD,
        "Mean RLSDCD4" => MeRLSDCD4,
        "Mean fRLSDCD1" => MefRLSDCD,
        "Mean fRLSDCD4" => MefRLSDCD4,
        "Min RLS" => MeRLS,
        "Min RLSDCD1" => MiRLSDCD,
        "Min RLSDCD4" => MiRLSDCD4,
        "Min fRLSDCD1" => MifRLSDCD,
        "Min fRLSDCD4" => MifRLSDCD4
    )
    CSV.write("ResultadosJuliaL100inbounds.csv", ResJulia)
    # @save "fRLSDCD.jld" MefRLSDCD MeRLSDCD MefRLSDCD4 MeRLSDCD4 MeRLS TfRLSDCD TRLSDCD TfRLSDCD4 TRLSDCD4 TRLS M L N
    # clf()
    # loglog(M,MeRLS,label="RLS")
    # loglog(M,MeRLSDCD,label="RLS-DCD Nu = 1")
    # loglog(M,MeRLSDCD4,label="RLS-DCD Nu = 4")
    # loglog(M,MefRLSDCD,label="fast RLS-DCD Nu = 1")
    # loglog(M,MefRLSDCD4,label="fast RLS-DCD Nu = 4")
    # legend()
    # grid()
    # xlabel("M")
    # title("Running time")
    # show()
    # savefig("fRLSDCD.svg")
end # test_DCBF

function test_ACBF()
    print("running test_ACBF!\n")
    ϵ=1.0
    σv=0.1
    λ = 0.99
    δ = 1.0

    L=100
    M=[10, 100, 1000] 
    N=5_000
    NM=length(M)
    TRLS=zeros(NM,L)
    TRLSDCD=zeros(NM,L)
    TRLSDCD4=zeros(NM,L)
    TfRLSDCD=zeros(NM,L)
    TfRLSDCD4=zeros(NM,L)

    for k=1:length(M)
        hi=randn(M[k])
        w=zeros(M[k])
        e=zeros(N)
        x=randn(N)
        s=filt(hi,1,x)+σv*randn(N)
        x=[zeros(M[k]);x]
        print("M= ", M[k], "\n")
        for i=1:L
            TfRLSDCD[k,i]=@elapsed fACBF_RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 1)
            TfRLSDCD4[k,i]=@elapsed fACBF_RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 4)
            TRLS[k,i]=@elapsed RLS!(w, e, hi, x, s, M[k], λ, δ)
            TRLSDCD[k,i]=@elapsed RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 1)
            TRLSDCD4[k,i]=@elapsed RLSDCD!(w, e, hi, x, s, M[k], λ, δ, 4)
            print("i= ", i, "\nTempo RLS\t\t", TRLS[k,i],
            "\nTempo RLSDCD Nu = 1\t", TRLSDCD[k,i],
            "\nTempo RLSDCD Nu = 4\t", TRLSDCD4[k,i],
            "\nTempo fRLSDCD Nu = 1\t", TfRLSDCD[k,i],
            "\nTempo fRLSDCD Nu = 4\t", TfRLSDCD4[k,i], "\n\n")
        end
        print("\n")
    end
    MefRLSDCD=mean(TfRLSDCD,dims=2)[:]
    MeRLSDCD=mean(TRLSDCD,dims=2)[:]
    MefRLSDCD4=mean(TfRLSDCD4,dims=2)[:]
    MeRLSDCD4=mean(TRLSDCD4,dims=2)[:]
    MeRLS=mean(TRLS,dims=2)[:]
    MifRLSDCD=minimum(TfRLSDCD,dims=2)[:]
    MiRLSDCD=minimum(TRLSDCD,dims=2)[:]
    MifRLSDCD4=minimum(TfRLSDCD4,dims=2)[:]
    MiRLSDCD4=minimum(TRLSDCD4,dims=2)[:]
    MiRLS=minimum(TRLS,dims=2)[:]
    ResJulia = DataFrame(
        "M" => M,
        "Mean RLS" => MeRLS,
        "Mean RLSDCD1" => MeRLSDCD,
        "Mean RLSDCD4" => MeRLSDCD4,
        "Mean fRLSDCD1" => MefRLSDCD,
        "Mean fRLSDCD4" => MefRLSDCD4,
        "Min RLS" => MeRLS,
        "Min RLSDCD1" => MiRLSDCD,
        "Min RLSDCD4" => MiRLSDCD4,
        "Min fRLSDCD1" => MifRLSDCD,
        "Min fRLSDCD4" => MifRLSDCD4
    )
    CSV.write("ResultadosJuliaL100ACBF.csv", ResJulia)
    # @save "fRLSDCD.jld" MefRLSDCD MeRLSDCD MefRLSDCD4 MeRLSDCD4 MeRLS TfRLSDCD TRLSDCD TfRLSDCD4 TRLSDCD4 TRLS M L N
    # clf()
    # loglog(M,MeRLS,label="RLS")
    # loglog(M,MeRLSDCD,label="RLS-DCD Nu = 1")
    # loglog(M,MeRLSDCD4,label="RLS-DCD Nu = 4")
    # loglog(M,MefRLSDCD,label="fast RLS-DCD Nu = 1")
    # loglog(M,MefRLSDCD4,label="fast RLS-DCD Nu = 4")
    # legend()
    # grid()
    # xlabel("M")
    # title("Running time")
    # show()
    # savefig("fRLSDCD.svg")
end # test_ACBF

function test_RLS(N, M)
    ϵ=1.0
    σv=0.01
    λ = 0.99
    δ = 1.0
    hi = randn(M)
    w = zeros(M)
    e = zeros(N)
    x = randn(N)
    xx = [zeros(M-1); x]
    s = filt(hi, 1, x) + σv * randn(N)
    RLS!(w, e, hi, xx, s, M, λ, δ)
    println("MSD = ", norm(w-hi)^2)
    return hi, w
end

function test_RLSDCD(N, M, Nu)
    ϵ=1.0
    σv=0.01
    λ = 0.99
    δ = 1.0
    hi = randn(M)
    w = zeros(M)
    e = zeros(N)
    x = randn(N)
    xx = [zeros(M-1); x]
    s = filt(hi, 1, x) + σv * randn(N)
    RLSDCD!(w, e, hi, xx, s, M, λ, δ, Nu)
    println("MSD = ", norm(w-hi)^2)
    return hi, w
end

function test_fRLSDCD(N, M, Nu)
    ϵ=1.0
    σv=0.01
    λ = 0.99
    δ = 1.0
    hi = randn(M)
    w = zeros(M)
    e = zeros(N)
    x = randn(N)
    xx = [zeros(M-1); x]
    s = filt(hi, 1, x) + σv * randn(N)
    fDCBF_RLSDCD!(w, e, hi, xx, s, M, λ, δ, Nu)
    println("MSD = ", norm(w-hi)^2)
    return hi, w
end

function test_fACBF_RLSDCD(N, M, Nu)
    ϵ=1.0
    σv=0.01
    λ = 0.99
    δ = 1.0
    hi = randn(M)
    w = zeros(M)
    e = zeros(N)
    x = randn(N)
    xx = [zeros(M-1); x]
    s = filt(hi, 1, x) + σv * randn(N)
    fACBF_RLSDCD!(w, e, hi, xx, s, M, λ, δ, Nu)
    println("MSD = ", norm(w-hi)^2)
    return hi, w
end

end #module