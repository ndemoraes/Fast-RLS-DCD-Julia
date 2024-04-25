export initialize_matrix
export update_matrix!
export update_vector!
export print_matrix



function initialize_matrix(size::Int64, delta::Float64)
    # initialize major diagonal to all "deltas", rest to zeros
    initialized = delta * diagm(0 => ones(size));
    return initialized
end



# ***********************************************************************
# update regular matrix by shifting down to the right by one column + row 
# (used in RLSDCD function)
# ***********************************************************************
# input: matrix, vector and scalar for update equation
# output: updated matrix
# ***********************************************************************

function update_matrix!(mtrx::Array{Float64,2}, 
                        vtr, 
                        scalar::Float64)

    first_col = @view mtrx[:,1]
	@inbounds for row=length(vtr)-1:-1:1    
        @inbounds for col=1:length(vtr)-1   
        		# shift elements down 1 cell and to the right 1 cell
                @inbounds mtrx[col+1,row+1] = mtrx[col,row]
        end
            # makes matrix symmetric
            @inbounds mtrx[1,row+1] = mtrx[row+1,1]    
    end
    @inbounds for row=length(vtr)-1:-1:1
        # define first column
        @inbounds mtrx[row+1,1] = scalar * first_col[row+1] + vtr[1] * vtr[row+1]
        # makes matrix symmetric
        @inbounds mtrx[1,row+1] = mtrx[row+1,1] 
    end
    @inbounds mtrx[1,1] = scalar * first_col[1] + vtr[1]^2

end	

# *****************************************************************************************
# multiply column of the regular matrix by a factor and add result to a vector 
# (used in DCD function)
# *****************************************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# *****************************************************************************************

function update_vector!(vtr::Array{Float64,1}, mtrx::Array{Float64,2}, scalar::Float64, col_numb::Int64)
    # mtrx[row, col]
	@inbounds for row=1:length(vtr)    
        @inbounds vtr[row] -= scalar * mtrx[row,col_numb]
    end
end


function print_matrix(mtrx::Array{Float64,2})
    # mtrx[row, col]
	for row=1:size(mtrx)[1]      # size(mtrx) returns a tuple (one for each dimension) and I only need the first element
        for col=1:size(mtrx)[1]  
        		print(mtrx[row,col], "  ")  
        end
        print("\n")
    end
 
end