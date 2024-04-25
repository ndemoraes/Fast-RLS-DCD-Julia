export initialize_JCBF_Matrix
export update_JCBF_Matrix!
export update_JCBF_Vector!


function initialize_JCBF_Matrix(size::Int64, delta::Float64)
	
    # initialize first diagonal to all "delta", rest of jmtrx to all zeros
    ijmtrx=Array{CircularBuffer{Float64},1}(undef,size)
    @inbounds ijmtrx[1]=CircularBuffer{Float64}(size)
    @inbounds append!(ijmtrx[1], delta * ones(size))

    @inbounds for diagonal=2:size
        @inbounds ijmtrx[diagonal]=CircularBuffer{Float64}(size-diagonal+1)
        @inbounds append!(ijmtrx[diagonal],zeros(size-diagonal+1))
    end
	
    return ijmtrx
end	




# ***************************************************************************
# update DIAGONAL JULIA-NATIVE CIRCULAR-BUFFER FORMAT MATRIX by shifting down 
# and to the right by one column and one row 
# (used in fJCBF_RLSDCD function)
# ***************************************************************************
# input: JCBF matrix, vector and scalar for update equation
# output: updated JCBF matrix
# ***************************************************************************

function update_JCBF_Matrix!(mtrx::Array{CircularBuffer{Float64},1}, 
                             vtr, 
                             scalar::Float64)

    @inbounds for i=1:(length(vtr))
        temp = @inbounds scalar * mtrx[i][1] + vtr[1] * vtr[i]
        @inbounds pushfirst!(mtrx[i],temp)
        #equivalent to β[i] = λrls * res[i] + e[n] * u[i]
    end
    
end	# update_JCBF_Matrix
    
    
    
# ********************************************************************************
# multiply column of the DIAGONAL JULIA-NATIVE CIRCULAR-BUFFER FORMAT MATRIX by a  
# factor and add result to a vector (used in fDCD function)
# ********************************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# *******************************************************************************

function update_JCBF_Vector!(vtr::Array{Float64,1}, 
                             mtrx::Array{CircularBuffer{Float64},1}, 
                             scalar::Float64, 
                             col_num::Int64)
    
    diagonal = col_num      # diagonal starts out as column in the original standard matrix format  

    @inbounds for i=1:col_num
        @inbounds vtr[i]=vtr[i]-scalar * mtrx[diagonal][i]
        diagonal -= 1
    end
    diagonal=2
    @inbounds for i = col_num + 1 : length(vtr)
        @inbounds vtr[i] = vtr[i] - scalar * mtrx[diagonal][col_num]
        diagonal += 1
    end

end
    