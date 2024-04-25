export initialize_DCBF_Matrix
export print_DCBF_Matrix
export update_DCBF_Matrix!
export update_DCBF_Vector!



# **************************************************************
# structure to store matrix in DIAGONAL, CIRCULAR-BUFFER format 
# ***************************************************************
# dmtrx - matrix in diagonal circular-buffer format (DCBF)
# cb_head - index to location of head of circular buffer in each diagonal column
# diag_num_elements - number of elements in diagonal column, and thus 
#                 also the number of columns in jagged array
# ***************************************************************

mutable struct DCBF_Matrix
	dmtrx::Array{Array{Float64,1},1} # dmtrx[diagonal][CB_element]
	cb_head::Array{Int64,1}
	diag_num_elements::Array{Int64,1}
end	



function initialize_DCBF_Matrix(size::Int64, delta::Float64)
	
    # initialize first diagonal to all "delta", rest of dmtrx to all zeros
	idmtrx=Array{Array{Float64,1},1}(undef,size)
	idmtrx[1]=delta * ones(size)
    @inbounds for j=2:size
        idmtrx[j]=zeros(size-j+1)
    end
	icb_head=Int64.(ones(size))
    idiag_num_elements=size:-1:1
    idcbm = DCBF_Matrix(idmtrx, icb_head, idiag_num_elements)
	
    return idcbm
end	


function initialize_DCBF_Matrix_with_RegularMatrix(size::Int64, regular_matrix::Array{Float64,2})
    # used by testing functions
	
    idmtrx=Array{Array{Float64,1},1}(undef,size)
    @inbounds idmtrx[1]=zeros(size)
    @inbounds for j=2:size
        idmtrx[j]=zeros(size-j+1)
    end
    
	@inbounds for row=1:size
		@inbounds for col=row:size   
    		diag = col - row + 1
        	idmtrx[diag][row] = regular_matrix[row,col]  
        end 
    end 
	icb_head=Int64.(ones(size))
    idiag_num_elements=size:-1:1
    idcbm = DCBF_Matrix(idmtrx, icb_head, idiag_num_elements)
	
    return idcbm
end	


function print_DCBF_Matrix(mtrx::DCBF_Matrix)
    for diagonal = 1:length(mtrx.dmtrx)
        println("diagonal: ", diagonal, ", values in CB: ", mtrx.dmtrx[diagonal])
    end

    println("\n")
	println("Indices of head of circular buffer in each diagonal: ", mtrx.cb_head)
	println("Number of elements in each diagonal", mtrx.diag_num_elements)

end	

# ****************************************************************
# update DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by shifting down 
# and to the right by one column and one row 
# (used in fRLSDCD function)
# ****************************************************************
# input: DCBF matrix, vector and scalar for update equation
# output: updated DCBF matrix
# ****************************************************************

function update_DCBF_Matrix!(mtrx::DCBF_Matrix, vtr, scalar::Float64)

    @inbounds for i=1:(length(vtr)) 

        # index of head of circular buffer before updating
        temp=mtrx.cb_head[i]

        # shift diagonal up by moving pointer within diagonal (mod is to make it wrap around)
        mtrx.cb_head[i]=mod1(mtrx.cb_head[i] - 1, mtrx.diag_num_elements[i]) 

        # update a single element
        mtrx.dmtrx[i][mtrx.cb_head[i]] = scalar * mtrx.dmtrx[i][temp] + vtr[1] * vtr[i]  

    end
    
end	# update_DCBF_Matrix
    
    
    
# *********************************************************************
# multiply column of the DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by a  
# factor and add result to a vector (used in fDCD function)
# *********************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# ********************************************************************

function update_DCBF_Vector!(vtr::Array{Float64,1}, mtrx::DCBF_Matrix, scalar::Float64, col_num::Int64)
    
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
        vtr[i] = vtr[i] - scalar * mtrx.dmtrx[diagonal][cb_element] 

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
    