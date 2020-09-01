# import packages and set the precision
import numpy as np
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(precision=5)

# computing the desired low-rank approximation by adding sufficient number of singular values
def compute_lower_rank_approx_via_SVD(data_matrix, desired_quality):
    u, s, v = np.linalg.svd(data_matrix)
    global current_approximant
    
    for i in range(1, len(s)+1):
        # compute low-rank approximation for different number of sigular values
        re_mat = np.dot(np.dot(u[:, :i], np.diag(s[:i])), v[:i, :])
        quality = np.linalg.norm(re_mat)/np.linalg.norm(data_matrix)
        # pick the proper number that fits the desired fidelity best
        if quality >= desired_quality:
            current_approximant = re_mat
            break
    
    return current_approximant



# this function divides the n x d data matrix into k-many "evenly divided, as closely as possible" 
# blocks. 
# for n*m matrix the splitting follows the rule as the code:
def compute_image_block(data_matrix, k):
    # get the size of given matrix, and compute the relation with k
    nrows = data_matrix.shape[0]
    ncols = data_matrix.shape[1]
    quo1 = int(nrows/k)
    rem1 = int(nrows%k)
    quo2 = int(ncols/k)
    rem2 = int(ncols%k)
    # initialize a tuple matrix
    image_block = np.zeros(shape=(k,k))
    image_block = np.array(list(zip(image_block.ravel(),image_block.ravel())), 
                            dtype=('i4,i4')).reshape(image_block.shape)
    
    # insert values    
    image_block[:rem1, :rem2] = tuple([quo1+1, quo2+1])
    image_block[:rem1, rem2:] = tuple([quo1+1, quo2])
    image_block[rem1:, :rem2] = tuple([quo1, quo2+1])
    image_block[rem1:, rem2:] = tuple([quo1, quo2])
        
    return image_block



# find the lower rank approximation for a given quality on each block of segmented data
# the "verbose" boolean variable is set to True if we want to print the shape of the segmented data
def get_approximation(data_matrix, k, quality, verbose):
    #first get the matrix blocks
    matrix_blocks = compute_image_block(data_matrix, k)
    
    #for every tuple, get the sub-matrix and store them all into a list
    comb_matrix = []
    for i in range(k):
        for j in range(k):
            dim = matrix_blocks[i, j]
            start_col  = sum([pair[1] for pair in matrix_blocks[0,:j]])
            start_row  = sum([pair[0] for pair in matrix_blocks[:i,0]])
            submat = np.matrix(data_matrix[int(start_row):int(start_row+dim[0]), int(start_col):int(start_col+dim[1])])
            comb_matrix.append(submat)
    
    # calculate the low-rank approximation of each sub-matrix and store
    blocked_data_matrix = []
    for i in range(k):
        for j in range(k):
            temp_mat = comb_matrix[i*k+j]
            if verbose:
                print('Shape of ({0}, {1}) block: {2}'.format(i+1, j+1, temp_mat.shape))
            approx = np.matrix(compute_lower_rank_approx_via_SVD(temp_mat, quality))
            blocked_data_matrix.append(approx)

    return reconstruct_data_from_image_block(blocked_data_matrix,k)



# this function takes the k x k image_block and reconstucts a single data_matrix from it
def reconstruct_data_from_image_block(image_block, k) :
    row_join = []
    for i in range(k):
        row_join.append(np.hstack(image_block[i*k:(i+1)*k]))
    
    data_matrix = np.vstack(row_join)
    
    return data_matrix


def convert_float64_to_uint8(A) :
    A = A/A.max()
    A = 255 * A
    return A.astype(np.uint8)


# this function "combines" the three color matrices (in the required order) to form a 3D
# array that can be rendered/viewed 
def reconstruct_image_from_RGB_64bit_matrices(red, blue, green) :
    reconstructed_image = cv2.merge([convert_float64_to_uint8(blue), 
                                     convert_float64_to_uint8(green), 
                                     convert_float64_to_uint8(red)])
    return reconstructed_image



# verifying the block reconstruction procedure
A = np.random.random((10,10))
B = get_approximation(A, 4, 0.99999999999999, True)
C = get_approximation(A, 4, 0.9, False)
print(np.allclose(A,B))
print(np.allclose(A,C))


IMAGE = 'image.jpeg'
image = cv2.imread(IMAGE)
# change the colorspace to make the colors in the image look right
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# split the image into blue-green-red components -- keep in mind that each of 
# these entries are 8-bit ints (i.e. have a value between 0 and 255)
blue_image = image[:,:,0]
green_image = image[:,:,1]
red_image = image[:,:,2]

quality = 0.99

# let us try k = 2, 3, 4, 5 and see how the image segmentation works out
fig = plt.figure(figsize=(20, 30))
image_index = 1
axs = fig.add_subplot(5,1, image_index)
fig.tight_layout()
plt.imshow(image)
axs.set_title('Original')
image_index = image_index + 1

for k in range(2,6) :
    b = get_approximation(blue_image, k, 1 - ((1-quality)/k), False)
    g = get_approximation(green_image, k, 1 - ((1-quality)/k), False)
    r = get_approximation(red_image, k, 1 - ((1-quality)/k), False)
    axs = fig.add_subplot(5,1, image_index)
    fig.tight_layout()
    reconstructed_image = reconstruct_image_from_RGB_64bit_matrices(r, b, g)
    plt.imshow(reconstructed_image)
    axs.set_title('Quality = ' + str(round(quality,5)) + '; #Segments =' + str(k))
    image_index = image_index + 1
    
plt.show()