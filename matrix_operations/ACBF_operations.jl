export initialize_ACBF_Matrix
export print_ACBF_Matrix
export update_ACBF_Matrix!
export update_ACBF_Vector!



# **************************************************************
# structure to store matrix in DIAGONAL, CIRCULAR-BUFFER format 
# ***************************************************************
# dmtrx - matrix in diagonal circular-buffer format (DCBF)
# cb_head - index to location of head of circular buffer in each diagonal column
# diag_num_elements - number of elements in diagonal column, and thus 
#                 also the number of columns in jagged array
# ***************************************************************

mutable struct ACBF_Matrix
	dmtrx::Array{Float64,1} # 1-D array representation of DCBF matrix
    top::Array{Int64,1}
	cb_head::Array{Int64,1}
	diag_num_elements::Array{Int64,1}
end	



function initialize_ACBF_Matrix(size::Int64, delta::Float64)
	
    # initialize first diagonal to all "delta", rest of dmtrx to all zeros
	idmtrx=zeros(size*(size+1)÷2)
	idmtrx[1:size].=delta * ones(size)
	icb_head=Int64.(ones(size))
    idiag_num_elements=size:-1:1
    itop = Array{Int64}(undef, size)
    index = 0
    for i in 1:size
        itop[i] = index
        index += idiag_num_elements[i]
    end

    idcbm = ACBF_Matrix(idmtrx, itop, icb_head, idiag_num_elements)
	
    return idcbm
end	

# ****************************************************************
# update DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by shifting down 
# and to the right by one column and one row 
# (used in fRLSDCD function)
# ****************************************************************
# input: DCBF matrix, vector and scalar for update equation
# output: updated DCBF matrix
# ****************************************************************

function update_ACBF_Matrix!(mtrx::ACBF_Matrix, vtr, scalar::Float64)

    @inbounds for i=1:(length(vtr)) 

        # index of head of circular buffer before updating
        temp=mtrx.top[i]+mtrx.cb_head[i]

        # shift diagonal up by moving pointer within diagonal (mod is to make it wrap around)
        mtrx.cb_head[i]=mod1(mtrx.cb_head[i] - 1, mtrx.diag_num_elements[i]) 

        # update a single element
        mtrx.dmtrx[temp] = scalar * mtrx.dmtrx[temp] + vtr[1] * vtr[i]  

    end
    
end	# update_ACBF_Matrix
    
    
    
# *********************************************************************
# multiply column of the DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by a  
# factor and add result to a vector (used in fDCD function)
# *********************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# ********************************************************************

function update_ACBF_Vector!(vtr::Array{Float64,1}, mtrx::ACBF_Matrix, scalar::Float64, col_num::Int64)
    
    diagonal = col_num          # diagonal starts out as column in the original standard matrix format
    diagonal_step=-1             # The algorithm steps through the diagonals until it reaches the first, 
                                 # then changes direction
    cb_offset=0                  # cb_offset is the offset from the head element in the circular buffer
                                 # inside the circular buffer for that diagonal
    kstep=1                       
    
    # iterate over elements in the vector
    @inbounds for i= 1 : (length(vtr))     

        # given the diagonal, select the circular buffer element   
        cb_element = mod1(mtrx.cb_head[diagonal] + cb_offset , mtrx.diag_num_elements[diagonal]) 
       
        # update the vector   
        vtr[i] = vtr[i] - scalar * mtrx.dmtrx[mtrx.top[diagonal]+cb_element] 

        # select the next column
        diagonal = diagonal + diagonal_step
        cb_offset = cb_offset + kstep
        if diagonal == 0
            diagonal = 2
            diagonal_step = +1
            cb_offset = col_num - 1
            kstep = 0
        end
    end

end
    