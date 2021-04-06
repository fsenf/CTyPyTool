import numpy as np






def extract_trainig_data(data, labels, indeces, hour = 0): 
    input_data = []
    output_data = []
    for variable in data.variables:
         if "bt" in variable:
            masked_channel = np.array(data[variable])[hour,indeces[0],indeces[1]].flatten() 
            input_data.append(masked_channel)
    input_data = np.array(input_data)
    sp = input_data.shape
    input_data = input_data.flatten().reshape(sp[1],sp[0], order='F')
    
    output_data = np.array(labels["CT"])[hour,indeces[0], indeces[1]].flatten()
    return input_data, output_data




def create_difference_vectors(data, keep_original_values = False):
    '''
    Transfroms a set of training vectors into a set of inter-value-difference-vectors.    
    
    
    Parameters
    ----------
    data : array-like
        array containing training vectors 
        
    keep_original_values : bool
        flag for determinig if the creation of difference-values is additive or replacive
    
    
    Returns
    -------
    np.array
        array containing difference vectors 

    '''
    new_data = []
    for vector in data:
        new_vector = list(vector) if(keep_original_values) else []
        for i in range(len(vector)):
            for j in range(i+1, len(vector)):
                new_vector.append(vector[i]-vector[j])
        new_data.append(new_vector)
    return np.array(new_data)



    




